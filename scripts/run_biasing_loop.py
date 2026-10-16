"""
run_biasing_loop.py

The property-biasing loop from Westermayr et al. (2022), with EQGAT-diff in
place of G-SchNet.

    generate -> filter -> score with oracle -> select mean +/- sigma
             -> fine-tune on the selection -> repeat

This is the part of the paper the rest of the pipeline exists to support.

PREREQUISITES
-------------
1. A trained EQGAT-diff checkpoint that produces molecules with NON-ZERO
   validity. The loop selects from valid molecules; if validity is 0% every
   iteration selects nothing and the loop stops. Check this first by sampling
   and reading the `validity` metric.
2. A trained oracle checkpoint from train_oracle.py.

USAGE
-----
    conda run --no-capture-output -n eqgatdiff python run_biasing_loop.py \\
        --repo /content/eqgat-diff \\
        --dataset-root /content/eqgat-diff/data/oe62 \\
        --start-ckpt /content/drive/MyDrive/eqgat_runs/oe62_chno/run0/last.ckpt \\
        --oracle /content/oracle/oracle_gap.pt \\
        --work-dir /content/biasing \\
        --target gap --direction min \\
        --n-loops 5 --n-sample 400 --finetune-epochs 10

--direction min  selects molecules BELOW mean - sigma  (small gap, small IP)
--direction max  selects molecules ABOVE mean + sigma  (large EA)

Scaled-down relative to Paper 1, which generated 100k-200k molecules per loop
over 7-11 loops. The mechanism is identical; only the numbers differ.
"""

import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import sys

import numpy as np
import torch
from ase.io import read as ase_read
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import SchNet

RDLogger.DisableLog("rdApp.*")


# --------------------------------------------------------------------------
# Step 1: generate
# --------------------------------------------------------------------------

def generate(args, ckpt, out_dir):
    """Sample molecules by calling the repo's own evaluation script."""
    os.makedirs(out_dir, exist_ok=True)
    cmd = (
        f"cd {args.repo}/eqgat_diff && "
        f"PYTHONPATH={args.repo}/eqgat_diff "
        f"python experiments/run_evaluation.py "
        f"--dataset oe62 "
        f"--dataset-root {args.dataset_root} "
        f"--model-path {ckpt} "
        f"--save-dir {out_dir} "
        f"--batch-size {args.sample_batch} "
        f"--ngraphs {args.n_sample} "
        f"--save-xyz"
    )
    print(f"\n[generate] {args.n_sample} molecules from {osp.basename(ckpt)}")
    # run_evaluation.py exits with an IndexError when zero molecules are
    # stable; the xyz files are written before that, so a non-zero exit code
    # is not necessarily fatal here. What *is* fatal is exiting before any
    # .xyz is written at all -- the caller checks for that.
    proc = subprocess.run(cmd, shell=True)
    return proc.returncode


# --------------------------------------------------------------------------
# Step 2: filter
# --------------------------------------------------------------------------

def xyz_to_mol(path, charges=(0, 1, -1, 2, -2)):
    """
    Read one generated .xyz. Returns (mol, full_bonds) or None.

    Two stages, because the two consumers need different things:

    Stage 1, connectivity only. Enough to reject fragmented structures, and
    enough for the oracle, which reads atomic numbers and coordinates and
    never looks at a bond.

    Stage 2, full bond orders. Only molecules that get this far can be written
    to an SDF for fine-tuning. Bond-order perception is tried at several net
    charges: insisting on charge=0 discards every generated ion, and that
    alone was throwing away most of the sample.
    """
    try:
        atoms = ase_read(path)
    except Exception:
        return None

    syms = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    def build():
        rw = Chem.RWMol()
        for sym in syms:
            rw.AddAtom(Chem.Atom(sym))
        conf = Chem.Conformer(rw.GetNumAtoms())
        for i, pos in enumerate(positions):
            conf.SetAtomPosition(i, Point3D(*(float(c) for c in pos)))
        m = rw.GetMol()
        m.AddConformer(conf, assignId=True)
        return m

    # ---- stage 1 --------------------------------------------------------
    base = build()
    try:
        rdDetermineBonds.DetermineConnectivity(base)
    except Exception:
        return None
    try:
        if len(Chem.GetMolFrags(base)) != 1:
            return None
    except Exception:
        return None

    # ---- stage 2 --------------------------------------------------------
    for q in charges:
        m = build()
        try:
            rdDetermineBonds.DetermineBonds(m, charge=q)
            Chem.SanitizeMol(m)
        except Exception:
            continue
        try:
            if len(Chem.GetMolFrags(m)) != 1:
                continue
        except Exception:
            continue
        return m, True

    return base, False


def collect_valid(sample_dir):
    """
    Every connected generated structure, each tagged with whether bond-order
    perception succeeded. Returns a list of (mol, full_bonds).
    """
    paths = sorted(glob.glob(osp.join(sample_dir, "**", "*.xyz"), recursive=True))
    print(f"[filter] {len(paths)} raw .xyz files found")

    items = []
    for p in paths:
        r = xyz_to_mol(p)
        if r is not None:
            items.append(r)

    n_full = sum(1 for _, f in items if f)

    def pct(n):
        return 100.0 * n / max(len(paths), 1)

    print(f"[filter] {len(items)} connected structures ({pct(len(items)):.1f}%) "
          f"-- scoreable by the oracle")
    print(f"[filter] {n_full} with full bond orders ({pct(n_full):.1f}%) "
          f"-- usable for fine-tuning")
    return items


# --------------------------------------------------------------------------
# Step 3: score with the oracle
# --------------------------------------------------------------------------

def load_oracle(path, device):
    ckpt = torch.load(path, map_location=device)
    model = SchNet(
        hidden_channels=ckpt["hidden"],
        num_filters=ckpt["hidden"],
        num_interactions=ckpt["interactions"],
        num_gaussians=50,
        cutoff=ckpt["cutoff"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    mae = float(ckpt.get("val_mae", float("nan")))
    print(f"[oracle] {osp.basename(path)}: target={ckpt['target']} "
          f"val_MAE={mae:.4f} eV")
    return model, ckpt["mean"], ckpt["std"], mae


def score(mols, oracle, mean, std, device, batch_size=32):
    """Predict the property for every molecule. Returns a numpy array."""
    data_list = []
    for m in mols:
        conf = m.GetConformers()[0]
        z = torch.tensor([a.GetAtomicNum() for a in m.GetAtoms()], dtype=torch.long)
        pos = torch.tensor(conf.GetPositions(), dtype=torch.float)
        pos = pos - pos.mean(dim=0, keepdim=True)
        data_list.append(Data(z=z, pos=pos))

    loader = DataLoader(data_list, batch_size=batch_size)
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = oracle(batch.z, batch.pos, batch.batch).view(-1)
            preds.append((out * std + mean).cpu())

    return torch.cat(preds).numpy()


# --------------------------------------------------------------------------
# Step 4: select
# --------------------------------------------------------------------------

def select(items, values, direction, x=1.0):
    """
    Paper 1's rule:
        min : keep  value <  mean - x*sigma
        max : keep  value >  mean + x*sigma

    `items` are (mol, full_bonds) pairs and are returned in the same form.
    """
    mean, sigma = float(np.mean(values)), float(np.std(values))
    if direction == "min":
        mask = values < mean - x * sigma
    else:
        mask = values > mean + x * sigma

    kept = [it for it, k in zip(items, mask) if k]
    print(f"[select] mean={mean:.3f} sigma={sigma:.3f} -> kept {len(kept)}/{len(items)}")
    if len(kept):
        print(f"[select] kept range: {values[mask].min():.3f} .. {values[mask].max():.3f}")
    return kept, mean, sigma


# --------------------------------------------------------------------------
# Step 5: rebuild the dataset and fine-tune
# --------------------------------------------------------------------------

def write_dataset(mols, dataset_root, seed=42):
    """Overwrite the raw SDFs with the selection and clear the PyG cache."""
    raw = osp.join(dataset_root, "raw")
    os.makedirs(raw, exist_ok=True)

    # Clearing processed/ is essential -- PyG silently reuses it otherwise,
    # and the fine-tune would run on the previous loop's molecules.
    processed = osp.join(dataset_root, "processed")
    if osp.isdir(processed):
        subprocess.run(f"rm -rf {processed}", shell=True)

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(mols))
    n_train = max(1, int(0.8 * len(mols)))
    n_val = max(1, int(0.1 * len(mols)))

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }
    # guard against an empty test split on very small selections
    if len(splits["test"]) == 0:
        splits["test"] = splits["val"]

    for name, ids in splits.items():
        path = osp.join(raw, f"oe62_{name}.sdf")
        w = Chem.SDWriter(path)
        for i in ids:
            w.write(mols[int(i)])
        w.close()
        print(f"[dataset] {len(ids):>5} -> {path}")


def warm_start(args, oracle, o_mean, o_std, device):
    """
    Paper 1's first biasing step, which does NOT use generated molecules.

    Westermayr et al. seed the loop by retraining the generator on the
    extreme decile of OE62 itself -- "a small subset of molecules in the
    OE62 data with property values below or above a certain threshold
    (about 10% of the OE62 data set)". That gives the first fine-tune
    thousands of real molecules in the target range, where selecting from
    a single batch of generated samples gives tens.

    Deviation to state: they threshold on the dataset's own property
    values; we score the training molecules with the surrogate instead,
    because the SDF written by the preparation step does not carry labels.
    At a surrogate MAE well below the width of a decile this changes which
    molecules sit near the boundary, not which decile they fall in.
    """
    raw = osp.join(args.dataset_root, "raw", "oe62_train.sdf")
    if not osp.exists(raw):
        raise SystemExit(f"--warm-start-frac needs the training set at {raw}")

    print("\n" + "=" * 68)
    print(f"WARM START  (Paper 1's first biasing step, "
          f"{100 * args.warm_start_frac:.0f}% of the training set)")
    print("=" * 68)
    print(f"[warm] reading {raw}")

    suppl = Chem.SDMolSupplier(raw, removeHs=False, sanitize=True)
    mols = [m for m in suppl if m is not None and m.GetNumConformers() > 0]
    if not mols:
        raise SystemExit(f"no readable molecules in {raw}")
    print(f"[warm] {len(mols)} training molecules read")

    values = score(mols, oracle, o_mean, o_std, device)
    k = max(1, int(round(args.warm_start_frac * len(mols))))
    order = np.argsort(values)
    idx = order[:k] if args.direction == "min" else order[-k:]
    kept = [mols[int(i)] for i in idx]
    sel = values[idx]

    print(f"[warm] {args.target}: full set mean={values.mean():.3f} "
          f"min={values.min():.3f} max={values.max():.3f}")
    print(f"[warm] keeping the {'lowest' if args.direction == 'min' else 'highest'} "
          f"{k} ({100.0 * k / len(mols):.1f}%), "
          f"range {sel.min():.3f} .. {sel.max():.3f}")

    write_dataset(kept, args.dataset_root)
    ckpt = finetune(args, args.start_ckpt,
                    osp.join(args.work_dir, "warm", "model"),
                    args.finetune_epochs)
    print(f"[warm] warm-started checkpoint: {ckpt}")
    return ckpt, {
        "stage": "warm_start",
        "n_training": len(mols),
        "frac": args.warm_start_frac,
        "n_selected": k,
        "full_mean": float(values.mean()),
        "selected_mean": float(sel.mean()),
        "selected_min": float(sel.min()),
        "selected_max": float(sel.max()),
    }


def finetune(args, from_ckpt, save_dir, epochs):
    os.makedirs(save_dir, exist_ok=True)
    cfg_path = osp.join(save_dir, "config.yaml")
    with open(cfg_path, "w") as f:
        f.write(f"""
dataset: oe62
dataset_root: {args.dataset_root}
save_dir: {save_dir}
id: 0

test_interval: 100000
num_epochs: {epochs}

sdim: {args.sdim}
vdim: {args.vdim}
num_layers: {args.num_layers}
edim: {args.edim}
rbf_dim: {args.rbf_dim}

batch_size: {args.batch_size}
inference_batch_size: {args.batch_size}
num_workers: 2
lr: {args.finetune_lr}

continuous: false
diffusion_pretraining: false
remove_hs: false
use_adaptive_loader: false
select_train_subset: false
""")

    cmd = (
        f"cd {args.repo}/eqgat_diff && "
        f"PYTHONPATH={args.repo}/eqgat_diff "
        f"python experiments/run_train.py --conf {cfg_path} "
        f"--load-ckpt {from_ckpt}"
    )
    print(f"\n[finetune] {epochs} epochs from {osp.basename(from_ckpt)}")
    subprocess.run(cmd, shell=True, check=True)

    out = osp.join(save_dir, "run0", "last.ckpt")
    if not osp.exists(out):
        raise SystemExit(f"fine-tuning produced no checkpoint at {out}")
    return out


# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="/content/eqgat-diff")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--start-ckpt", required=True)
    p.add_argument("--oracle", required=True)
    p.add_argument("--work-dir", default="/content/biasing")

    p.add_argument("--target", default="gap")
    p.add_argument("--direction", default="min", choices=["min", "max"])
    p.add_argument("--sigma-x", type=float, default=1.0)

    p.add_argument("--n-loops", type=int, default=5)
    p.add_argument("--n-sample", type=int, default=400)
    p.add_argument("--sample-batch", type=int, default=20)
    p.add_argument("--finetune-epochs", type=int, default=10)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--min-selected", type=int, default=20,
                   help="Abort if a loop selects fewer than this")
    p.add_argument("--warm-start-frac", type=float, default=0.0,
                   help="Before iteration 0, fine-tune on this fraction of "
                        "the TRAINING set at the target extreme, as Paper 1 "
                        "does with the extreme decile of OE62 (use 0.1). "
                        "0 disables it. Ignored when --loop-offset > 0, so "
                        "a resumed run does not warm-start twice.")
    p.add_argument("--oracle2", default=None,
                   help="Second, independently seeded oracle checkpoint. "
                        "Enables Paper 1's step 6: molecules whose two "
                        "predictions disagree by more than the models' "
                        "combined MAE are discarded before selection, and the "
                        "survivors are scored with the mean of the two.")
    p.add_argument("--agree-tol", type=float, default=None,
                   help="Override the disagreement threshold in eV. Default "
                        "is the sum of the two oracles' validation MAEs.")
    p.add_argument("--select-from", default="bonded",
                   choices=["bonded", "connected"],
                   help="Which pool the selection is made over. 'bonded' "
                        "(default) uses only molecules whose bond orders were "
                        "perceived, matching Paper 1's validity filter. "
                        "'connected' scores every single-component structure, "
                        "which is a larger pool but a deviation worth stating "
                        "-- the oracle reads coordinates only, so it can score "
                        "them, but they were not validated as chemistry.")
    p.add_argument("--loop-offset", type=int, default=0,
                   help="Number the first loop of this run from here instead "
                        "of 0. Use it to continue a loop sequence in a new "
                        "Kaggle commit: pass --start-ckpt pointing at the "
                        "last loop's model and --loop-offset set to the next "
                        "loop number, so existing loopN/ directories and "
                        "history.json entries are preserved rather than "
                        "overwritten.")

    # must match the generator's architecture
    p.add_argument("--sdim", type=int, default=128)
    p.add_argument("--vdim", type=int, default=32)
    p.add_argument("--num-layers", type=int, default=6)
    p.add_argument("--edim", type=int, default=32)
    p.add_argument("--rbf-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)

    args = p.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    oracle, o_mean, o_std, o_mae = load_oracle(args.oracle, device)

    # Paper 1 step 6: a second, independently seeded surrogate, with any
    # molecule the two disagree on discarded before selection.
    oracle2 = o2_mean = o2_std = None
    if args.oracle2:
        oracle2, o2_mean, o2_std, o2_mae = load_oracle(args.oracle2, device)
        agree_tol = (args.agree_tol if args.agree_tol is not None
                     else o_mae + o2_mae)
        print(f"[oracle] agreement filter on: discard if the two predictions "
              f"differ by more than {agree_tol:.4f} eV")
    else:
        agree_tol = None
        print("[oracle] single surrogate -- no agreement filter "
              "(pass --oracle2 to enable Paper 1's step 6)")

    ckpt = args.start_ckpt
    warm_record = None
    if args.warm_start_frac > 0 and args.loop_offset == 0:
        ckpt, warm_record = warm_start(args, oracle, o_mean, o_std, device)

    # Continuing an earlier commit: keep the history already on disk so the
    # final summary covers every loop, not just the ones in this run.
    hist_path = osp.join(args.work_dir, "history.json")
    history = []
    if args.loop_offset > 0 and osp.exists(hist_path):
        try:
            with open(hist_path) as f:
                history = [h for h in json.load(f)
                           if h.get("loop", 0) < args.loop_offset]
            print(f"[resume] carried {len(history)} earlier loop(s) forward "
                  f"from {hist_path}")
        except Exception as e:
            print(f"[resume] could not read {hist_path}: {e}")

    for k in range(args.n_loops):
        loop = k + args.loop_offset
        print("\n" + "=" * 68)
        print(f"LOOP {loop}")
        print("=" * 68)

        sample_dir = osp.join(args.work_dir, f"loop{loop}", "samples")
        rc = generate(args, ckpt, sample_dir)

        # Distinguish "the sampler crashed" from "the sampler ran but its
        # output was all invalid". Both end with zero usable molecules, but
        # they need completely different fixes, and guessing wrong wastes
        # hours in an unattended run.
        n_xyz = len(glob.glob(osp.join(sample_dir, "**", "*.xyz"),
                              recursive=True))
        if n_xyz == 0:
            raise SystemExit(
                f"\nGeneration produced no .xyz files at all "
                f"(run_evaluation.py exited {rc}).\n"
                "This is a crash in the sampler, NOT a weak generator -- read "
                "the traceback above.\n"
                "The usual cause is a stale oe62_dataset.py whose "
                "OE62DataModule does not set self.train_dataset, which "
                "run_evaluation.py reads when built with only_stats=True. "
                "Check with:\n"
                "    grep -n 'self.train_dataset' "
                ".../experiments/data/oe62/oe62_dataset.py\n"
                "If that prints nothing, patch the file and delete its "
                "__pycache__ before retrying."
            )

        items = collect_valid(sample_dir)
        if len(items) == 0:
            raise SystemExit(
                f"\n{n_xyz} structures were generated, but none was a single "
                "connected component.\n"
                "The sampler works; its output is fragmented. The generator "
                "needs more training -- sample it directly with "
                "run_evaluation.py and check `connected_components`."
            )

        # Record the funnel before any narrowing, so history.json shows where
        # the molecules actually went.
        n_connected = len(items)
        n_bond_complete = sum(1 for _, f in items if f)

        if args.select_from == "bonded":
            pool = [it for it in items if it[1]]
            print(f"[pool] selecting over {len(pool)} bond-complete molecules "
                  f"(Paper 1's validity filter)")
        else:
            pool = items
            print(f"[pool] selecting over all {len(pool)} connected structures")

        if len(pool) == 0:
            raise SystemExit(
                f"\n{len(items)} connected structures, but none has usable "
                "bond orders, so there is nothing to select from.\n"
                "Either the generator needs more training, or rerun with "
                "--select-from connected to score raw geometries instead."
            )

        items = pool
        mols = [m for m, _ in items]
        values = score(mols, oracle, o_mean, o_std, device)

        n_before_agree = len(values)
        if oracle2 is not None:
            values2 = score(mols, oracle2, o2_mean, o2_std, device)
            spread = np.abs(values - values2)
            keep = spread <= agree_tol
            n_drop = int((~keep).sum())
            print(f"[agree] |pred1 - pred2|: mean={spread.mean():.3f} "
                  f"max={spread.max():.3f} eV")
            print(f"[agree] discarded {n_drop}/{n_before_agree} molecules the "
                  f"two surrogates disagree on "
                  f"({100.0 * n_drop / max(n_before_agree, 1):.1f}%)")
            if keep.sum() == 0:
                raise SystemExit(
                    "\nThe two surrogates disagree on every molecule. Either "
                    "one checkpoint is broken, or the generated structures "
                    "are far enough off-distribution that neither model is "
                    "usable. Check both oracles' val MAE before continuing."
                )
            items = [it for it, k in zip(items, keep) if k]
            # The consensus value is the mean of the two surviving predictions.
            values = 0.5 * (values[keep] + values2[keep])

        print(f"[score] {args.target}: mean={values.mean():.3f} "
              f"min={values.min():.3f} max={values.max():.3f}")

        kept, mean, sigma = select(items, values, args.direction, args.sigma_x)

        # Scoring needs coordinates only; fine-tuning needs bond orders. So
        # the selection is made over everything the oracle could score, and
        # only the bond-complete subset of it reaches the training set.
        trainable = [m for m, full in kept if full]
        if len(kept):
            print(f"[select] {len(trainable)}/{len(kept)} of the selection has "
                  f"full bond orders and can be fine-tuned on")

        history.append({
            "loop": loop,
            "n_generated": args.n_sample,
            "n_xyz": n_xyz,
            "n_connected": n_connected,
            "n_bond_complete": n_bond_complete,
            "n_scored": n_before_agree,
            "n_after_agreement": len(items),
            "n_valid": len(items),
            "mean": mean,
            "sigma": sigma,
            "min": float(values.min()),
            "max": float(values.max()),
            "n_selected": len(kept),
            "n_trainable": len(trainable),
        })
        with open(osp.join(args.work_dir, "history.json"), "w") as f:
            json.dump(([warm_record] if warm_record else []) + history,
                      f, indent=2)

        if len(trainable) < args.min_selected:
            print(f"\nOnly {len(trainable)} selected molecules have full bond "
                  f"orders (need {args.min_selected}). Stopping: fine-tuning "
                  f"on this few would overfit badly.\n"
                  f"Raise --n-sample, lower --min-selected, or widen "
                  f"--sigma-x.")
            break

        write_dataset(trainable, args.dataset_root)
        ckpt = finetune(
            args, ckpt,
            osp.join(args.work_dir, f"loop{loop}", "model"),
            args.finetune_epochs,
        )

    print("\n" + "=" * 68)
    print("SUMMARY")
    print("=" * 68)
    print(f"{'loop':>5} {'conn':>6} {'bonded':>7} {'mean':>9} {'min':>9} "
          f"{'sel':>5} {'train':>6}")
    for h in history:
        print(f"{h['loop']:>5} {h.get('n_connected', h['n_valid']):>6} "
              f"{h.get('n_bond_complete', -1):>7} {h['mean']:>9.3f} "
              f"{h['min']:>9.3f} {h['n_selected']:>5} "
              f"{h.get('n_trainable', -1):>6}")
    print(f"\nhistory written to {osp.join(args.work_dir, 'history.json')}")
    print("\nThe number to report is how far `mean` moves across loops, and "
          "how far `min` (or `max`) goes beyond the range present in OE62.")


if __name__ == "__main__":
    main()
