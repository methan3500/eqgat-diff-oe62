"""
train_oracle.py

Trains a property oracle on OE62: 3D structure -> electronic property
(fundamental gap dE, ionization potential IP, or electron affinity EA).

This replaces the SchNet+H weights from Paper 1, which the authors did not
publish in a usable form.

WHY THIS IS BUILT ON PyG RATHER THAN SchNarc / schnetpack 0.3
--------------------------------------------------------------
Paper 1's oracle is SchNet+H, built on the 2019 schnetpack 0.3 stack. Two
reasons not to resurrect that here:

1. Classifier guidance (the "Variant B" biasing mechanism, and the one the
   EQGAT-diff repo already has hooks for -- see run_evaluation.py's
   --use-energy-guidance / --ckpt-energy-model / --guidance-scale flags)
   requires the oracle's GRADIENTS during reverse diffusion sampling. That
   means the oracle must live in the same process, and therefore the same
   environment, as EQGAT-diff. A schnetpack-0.3 model in a separate Python
   3.7 environment cannot supply those gradients.

2. torch_geometric ships SchNet as a built-in model, so the same
   architecture family is available in the eqgatdiff environment with no
   extra install and no version archaeology.

The trade-off, which belongs in your write-up: Paper 1's SchNet+H predicts
a whole eigenvalue spectrum and derives HOMO/LUMO from it, and applies a
delta-learning correction from PBE0 to G0W0@PBE0. This trains a direct
regressor on whatever target label OE62 provides. It is a simplification,
and its accuracy must be measured (reported below as validation MAE) rather
than assumed to match Paper 1's ~0.25 eV.

USAGE
-----
Run inspect_oe62.py FIRST to learn the label names, then either:

  # labels stored as scalars in key_value_pairs or data
  conda run -n eqgatdiff python train_oracle.py \\
      --db /content/data/train.db --label-key gap --limit 10000

  # labels stored as an eigenvalue array (Paper 1 style: HOMO at index 50)
  conda run -n eqgatdiff python train_oracle.py \\
      --db /content/data/train.db --eig-key eigenvalues_pbe0 \\
      --homo-index 50 --target gap --limit 10000
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from ase.db import connect
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet
from tqdm import tqdm


# --------------------------------------------------------------------------
# Label extraction
# --------------------------------------------------------------------------

def get_stored(row, key):
    """Look for `key` in key_value_pairs first, then data. None if absent."""
    kvp = row.key_value_pairs or {}
    if key in kvp:
        return kvp[key]
    try:
        data = row.data or {}
    except Exception:
        return None
    if key in data:
        return data[key]
    return None


def label_from_row(row, args):
    """
    Return the scalar target for one row, or None if it cannot be computed.

    Two modes:
      --label-key   : read one stored scalar directly
      --eig-key     : read an eigenvalue array, take HOMO/LUMO from it
    """
    if args.label_key is not None:
        val = get_stored(row, args.label_key)
        if val is None:
            return None
        try:
            val = float(np.asarray(val).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return None
        # OE62 stores the HOMO-LUMO gap under 'hlgap' with a negative sign.
        # --negate flips it so the oracle regresses a positive gap in eV.
        return -val if args.negate else val

    eig = get_stored(row, args.eig_key)
    if eig is None:
        return None
    eig = np.asarray(eig).ravel()
    if eig.size <= args.homo_index + 1:
        return None

    eig = np.sort(eig)
    homo = float(eig[args.homo_index])
    lumo = float(eig[args.homo_index + 1])

    if args.target == "gap":
        return abs(lumo - homo)
    if args.target == "homo":
        return homo
    if args.target == "lumo":
        return lumo
    # Koopmans-style identification, as used in Paper 1
    if args.target == "ip":
        return -homo
    if args.target == "ea":
        return -lumo
    return None


def build_dataset(args):
    db = connect(args.db)
    total = db.count()
    print(f"database holds {total} rows")

    allowed = None
    if args.elements:
        allowed = {e.strip() for e in args.elements.split(",") if e.strip()}
        print(f"element whitelist: {sorted(allowed)}")

    data_list = []
    n_no_label = 0
    n_too_big = 0
    n_bad_element = 0

    rows = db.select()
    for i, row in enumerate(tqdm(rows, total=min(total, args.limit or total),
                                 desc="Reading")):
        if args.limit is not None and len(data_list) >= args.limit:
            break

        atoms = row.toatoms()
        if len(atoms) > args.max_atoms:
            n_too_big += 1
            continue

        if allowed is not None and not set(
            atoms.get_chemical_symbols()
        ).issubset(allowed):
            n_bad_element += 1
            continue

        y = label_from_row(row, args)
        if y is None or not np.isfinite(y):
            n_no_label += 1
            continue

        z = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
        pos = torch.tensor(atoms.get_positions(), dtype=torch.float)
        pos = pos - pos.mean(dim=0, keepdim=True)

        data_list.append(Data(z=z, pos=pos, y=torch.tensor([y], dtype=torch.float)))

    print(f"\nusable molecules : {len(data_list)}")
    print(f"skipped, too big : {n_too_big}")
    print(f"skipped, element not in whitelist: {n_bad_element}")
    print(f"skipped, no label: {n_no_label}")

    if len(data_list) == 0:
        raise SystemExit(
            "\nNo labelled molecules found.\n"
            "Run inspect_oe62.py and check the label name you passed. If the\n"
            "database carries no electronic-structure values at all, an oracle\n"
            "cannot be trained from this file -- you need the labelled OE62\n"
            "release (Stuke et al. 2020) instead."
        )

    return data_list


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def run_epoch(model, loader, device, mean, std, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()

    total_abs_err = 0.0
    n_seen = 0

    for batch in loader:
        batch = batch.to(device)
        target = (batch.y - mean) / std

        with torch.set_grad_enabled(train):
            pred = model(batch.z, batch.pos, batch.batch).view(-1)
            loss = F.l1_loss(pred, target)

        if train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()

        # report error in the original units (eV), not normalized units
        total_abs_err += (loss.item() * std) * batch.num_graphs
        n_seen += batch.num_graphs

    return total_abs_err / max(n_seen, 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True)
    p.add_argument("--out-dir", default="/content/oracle")

    # label source: one of these two paths
    p.add_argument("--label-key", default=None,
                   help="Name of a stored scalar to regress directly")
    p.add_argument("--eig-key", default=None,
                   help="Name of a stored eigenvalue array")
    p.add_argument("--homo-index", type=int, default=50,
                   help="Index of HOMO in the sorted spectrum (Paper 1: 50)")
    p.add_argument("--target", default="gap",
                   choices=["gap", "homo", "lumo", "ip", "ea"])
    p.add_argument("--negate", action="store_true",
                   help="Flip the sign of --label-key values. OE62 stores "
                        "'hlgap' negative, so use this with --label-key hlgap.")

    p.add_argument("--limit", type=int, default=10000)
    p.add_argument("--max-atoms", type=int, default=50)
    p.add_argument("--elements", default=None,
                   help="Comma-separated whitelist, e.g. C,H,N,O. Match this "
                        "to the generator's training chemistry.")
    p.add_argument("--val-frac", type=float, default=0.1)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--interactions", type=int, default=6)
    p.add_argument("--cutoff", type=float, default=10.0)

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if (args.label_key is None) == (args.eig_key is None):
        raise SystemExit("Pass exactly one of --label-key or --eig-key.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data_list = build_dataset(args)

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(data_list))
    n_val = max(1, int(args.val_frac * len(data_list)))
    val_set = [data_list[i] for i in idx[:n_val]]
    train_set = [data_list[i] for i in idx[n_val:]]
    print(f"train / val : {len(train_set)} / {len(val_set)}")

    ys = torch.cat([d.y for d in train_set])
    mean, std = ys.mean().item(), ys.std().item()
    if std < 1e-8:
        raise SystemExit("Target has no variance -- check the label you chose.")
    print(f"target '{args.target}' : mean {mean:.3f}, std {std:.3f}, "
          f"range [{ys.min():.3f}, {ys.max():.3f}]")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SchNet(
        hidden_channels=args.hidden,
        num_filters=args.hidden,
        num_interactions=args.interactions,
        num_gaussians=50,
        cutoff=args.cutoff,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"oracle parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-6
    )

    ckpt_path = os.path.join(args.out_dir, f"oracle_{args.target}.pt")
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mae = run_epoch(model, train_loader, device, mean, std, optimizer)
        val_mae = run_epoch(model, val_loader, device, mean, std)
        scheduler.step(val_mae)

        flag = ""
        if val_mae < best_val:
            best_val = val_mae
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "mean": mean,
                    "std": std,
                    "target": args.target,
                    "hidden": args.hidden,
                    "interactions": args.interactions,
                    "cutoff": args.cutoff,
                    "val_mae": val_mae,
                },
                ckpt_path,
            )
            flag = "  <- saved"

        if epoch % 5 == 0 or epoch == 1 or flag:
            print(f"epoch {epoch:3d} | train MAE {train_mae:.4f} eV | "
                  f"val MAE {val_mae:.4f} eV{flag}")

    print(f"\nbest validation MAE: {best_val:.4f} eV")
    print(f"checkpoint: {ckpt_path}")
    print("\nFor reference, Paper 1 reports ~0.25 eV MAE for SchNet+H on OE62.")
    print("If this is far worse, the oracle is too weak to drive a biasing")
    print("loop -- train longer, on more molecules, or with a larger model")
    print("before trusting its selections.")


if __name__ == "__main__":
    main()
