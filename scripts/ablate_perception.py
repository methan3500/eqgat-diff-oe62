#!/usr/bin/env python
"""
ablate_perception.py

How many usable molecules does each bond-perception strategy recover from
the same set of generated structures?

This is the ablation behind the claim that the biasing loop's bottleneck is
bond perception rather than generation. It runs on .xyz files a sampler has
already written, so it needs no GPU and no re-sampling -- point it at a
loopN/samples directory from a previous commit.

Strategies compared:

  neutral        DetermineBonds(charge=0), then sanitize, then require a
                 single fragment. This is the obvious implementation and
                 the one most pipelines use.
  charge-search  DetermineBonds at net charge 0, +1, -1, +2, -2, first
                 success wins. Generated structures include ions, and
                 forcing neutrality discards all of them.
  connectivity   DetermineConnectivity only, then require a single
                 fragment. Bond ORDERS are not resolved, so these
                 molecules cannot be written to an SDF for fine-tuning --
                 but a surrogate that reads atomic numbers and coordinates
                 can still score them.

Usage:
  python ablate_perception.py --samples DIR [DIR ...] --names a,b
  python ablate_perception.py --samples DIR --timeout 5 --out ablation.json
"""

import argparse
import glob
import json
import os
import os.path as osp
import sys
import time

import numpy as np
from ase.io import read as ase_read
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

CHARGES = (0, 1, -1, 2, -2)


def load_xyz(path):
    """Return (symbols, positions) or None."""
    try:
        atoms = ase_read(path)
    except Exception:
        return None
    return atoms.get_chemical_symbols(), atoms.get_positions()


def build(syms, positions):
    rw = Chem.RWMol()
    for s in syms:
        rw.AddAtom(Chem.Atom(s))
    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, p in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(*(float(c) for c in p)))
    m = rw.GetMol()
    m.AddConformer(conf, assignId=True)
    return m


def one_fragment(mol):
    try:
        return len(Chem.GetMolFrags(mol)) == 1
    except Exception:
        return False


# --------------------------------------------------------------------------
# the three strategies
# --------------------------------------------------------------------------

def strat_neutral(syms, pos):
    m = build(syms, pos)
    try:
        rdDetermineBonds.DetermineBonds(m, charge=0)
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return m if one_fragment(m) else None


def strat_charge_search(syms, pos):
    for q in CHARGES:
        m = build(syms, pos)
        try:
            rdDetermineBonds.DetermineBonds(m, charge=q)
            Chem.SanitizeMol(m)
        except Exception:
            continue
        if one_fragment(m):
            return m
    return None


def strat_connectivity(syms, pos):
    m = build(syms, pos)
    try:
        rdDetermineBonds.DetermineConnectivity(m)
    except Exception:
        return None
    return m if one_fragment(m) else None


STRATEGIES = [
    ("neutral", strat_neutral,
     "DetermineBonds(charge=0), single fragment"),
    ("charge-search", strat_charge_search,
     "DetermineBonds at charge 0,+-1,+-2, single fragment"),
    ("connectivity", strat_connectivity,
     "DetermineConnectivity only, single fragment"),
]


def net_charge(mol):
    try:
        return sum(a.GetFormalCharge() for a in mol.GetAtoms())
    except Exception:
        return 0


def run_set(sample_dir, verbose=True):
    paths = sorted(glob.glob(osp.join(sample_dir, "**", "*.xyz"),
                             recursive=True))
    if not paths:
        print(f"  no .xyz found under {sample_dir}", file=sys.stderr)
        return None

    loaded = []
    n_unreadable = 0
    for p in paths:
        r = load_xyz(p)
        if r is None:
            n_unreadable += 1
        else:
            loaded.append(r)

    out = {
        "dir": sample_dir,
        "n_files": len(paths),
        "n_unreadable": n_unreadable,
        "strategies": {},
    }

    for name, fn, desc in STRATEGIES:
        t0 = time.time()
        kept = 0
        charged = 0
        for syms, pos in loaded:
            m = fn(syms, pos)
            if m is not None:
                kept += 1
                if net_charge(m) != 0:
                    charged += 1
        dt = time.time() - t0
        out["strategies"][name] = {
            "description": desc,
            "kept": kept,
            "share_of_files": kept / max(len(paths), 1),
            "charged_species": charged,
            "seconds": dt,
        }
        if verbose:
            print(f"    {name:14s} {kept:5d} / {len(paths):5d} "
                  f"({100.0 * kept / max(len(paths), 1):5.1f}%)  "
                  f"{charged:4d} charged   {dt:6.1f}s")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", nargs="+", required=True,
                   help="One or more directories of generated .xyz files")
    p.add_argument("--names", default=None,
                   help="Comma-separated labels, one per --samples entry")
    p.add_argument("--out", default="ablation.json")
    args = p.parse_args()

    names = (args.names.split(",") if args.names
             else [osp.basename(osp.dirname(d.rstrip("/"))) or f"set{i}"
                   for i, d in enumerate(args.samples)])
    if len(names) != len(args.samples):
        raise SystemExit("--names must have one entry per --samples entry")

    results = {}
    for name, d in zip(names, args.samples):
        print(f"\n{name}: {d}")
        r = run_set(d)
        if r is not None:
            results[name] = r

    if not results:
        raise SystemExit("nothing analysed")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)

    # ---- a table ready to paste into the paper --------------------------
    print("\n" + "=" * 72)
    print("LaTeX table")
    print("=" * 72)
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    print(r"Set & Strategy & Recovered & Share & Charged \\")
    print(r"\midrule")
    for name, r in results.items():
        first = True
        for sname, s in r["strategies"].items():
            label = name if first else ""
            first = False
            print(f"{label} & {sname} & {s['kept']} & "
                  f"\\SI{{{100.0 * s['share_of_files']:.1f}}}{{\\percent}} & "
                  f"{s['charged_species']} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ---- the headline ratio --------------------------------------------
    print("\n" + "=" * 72)
    for name, r in results.items():
        n = r["strategies"]["neutral"]["kept"]
        c = r["strategies"]["charge-search"]["kept"]
        k = r["strategies"]["connectivity"]["kept"]
        if n:
            print(f"{name}: charge search recovers {c / n:.2f}x the "
                  f"molecules of neutral-only; connectivity-only "
                  f"{k / n:.2f}x.")
        else:
            print(f"{name}: neutral-only recovered nothing; charge search "
                  f"{c}, connectivity {k}.")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
