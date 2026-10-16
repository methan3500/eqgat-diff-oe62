"""
analyze_generated.py

Implements the analysis Paper 1 (Westermayr et al.) performs in its section 2.3
— "Identification of bonding patterns" — so the same questions can be asked of
molecules produced by a diffusion generator.

WHAT PAPER 1 DID, AND WHAT THIS REPRODUCES
-------------------------------------------
After biasing G-SchNet toward small fundamental gaps, they asked *which
chemistry* the model had discovered. Their procedure:

  1. compute descriptors for the generated molecules and for OE62
  2. reduce with PCA, cluster (BIRCH + agglomerative)
  3. inspect cluster centroids for recurring bonding patterns
  4. they found cyano groups (C#N) strongly enriched in low-gap molecules
  5. knock-out test: delete every C#N from the training set, retrain, and see
     whether the bias rediscovers them

This script covers steps 1-4 and provides the counting needed for step 5.

Paper 1 used SOAP descriptors (structural) plus a bonding descriptor. SOAP
needs `dscribe` and 3D coordinates; here the structural role is filled by
Morgan fingerprints, which capture bonding environments directly and need no
extra dependency. The conclusions of interest — which functional groups are
enriched — do not depend on that choice, but it should be stated when
reporting.

Synthetic complexity: Paper 1 used SCScore (Coley et al., trained on Reaxys).
That model is not redistributable, so this uses RDKit's SA_Score, which is
built in and correlates well. Again: state which one you used.

USAGE
-----
    python analyze_generated.py \\
        --reference /path/oe62_train.sdf \\
        --generated /path/samples/ep100 \\
        --out-dir /path/analysis

--generated accepts a directory of .xyz files (EQGAT-diff's --save-xyz
output), a directory of .sdf files, or a single .sdf. Several --generated
arguments can be given to compare biasing iterations:

    --generated loop0_dir --generated loop1_dir --generated loop2_dir
"""

import argparse
import json
import os
import os.path as osp
import sys
from collections import Counter

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------
# Functional groups. The first is the one Paper 1 found enriched in low-gap
# molecules; the rest give context and cover the chemistry OE62 contains.
# --------------------------------------------------------------------------

SMARTS = {
    "nitrile_CN":        "[CX2]#[NX1]",
    "nitro":             "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "carbonyl":          "[CX3]=[OX1]",
    "carboxyl":          "[CX3](=O)[OX2H1]",
    "hydroxyl":          "[OX2H]",
    "amine":             "[NX3;H2,H1;!$(NC=O)]",
    "amide":             "[NX3][CX3](=[OX1])",
    "ether":             "[OD2]([#6])[#6]",
    "aromatic_ring":     "a1aaaaa1",
    "aromatic_5ring":    "a1aaaa1",
    "thiophene":         "c1ccsc1",
    "furan":             "c1ccoc1",
    "pyridine":          "c1ccncc1",
    "C=C":               "[CX3]=[CX3]",
    "C#C":               "[CX2]#[CX2]",
    "sulfone":           "[$([SX4](=O)(=O))]",
    "thioether":         "[#16X2]([#6])[#6]",
    "halogen_F":         "[F]",
    "halogen_Cl":        "[Cl]",
    "halogen_Br":        "[Br]",
    "halogen_I":         "[I]",
    "phosphate_like":    "[PX4]",
    "silicon":           "[Si]",
}

PATTERNS = {name: Chem.MolFromSmarts(s) for name, s in SMARTS.items()}
_BAD = [n for n, p in PATTERNS.items() if p is None]
if _BAD:
    raise SystemExit(f"bad SMARTS for: {_BAD}")


def sa_scorer():
    """RDKit's synthetic accessibility score, if the contrib module is there."""
    try:
        from rdkit.Chem import RDConfig
        sys.path.append(osp.join(RDConfig.RDContribDir, "SA_Score"))
        import sascorer  # noqa
        return sascorer.calculateScore
    except Exception:
        return None


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def mol_from_xyz(path):
    """Read an .xyz file and perceive bonds, as the biasing loop does."""
    try:
        with open(path) as fh:
            lines = fh.read().strip().splitlines()
        n = int(lines[0].split()[0])
        symbols, coords = [], []
        for line in lines[2:2 + n]:
            parts = line.split()
            symbols.append(parts[0])
            coords.append([float(x) for x in parts[1:4]])
    except Exception:
        return None

    rw = Chem.RWMol()
    for s in symbols:
        rw.AddAtom(Chem.Atom(s))
    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, c in enumerate(coords):
        conf.SetAtomPosition(i, Point3D(*c))
    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    if len(Chem.GetMolFrags(mol)) != 1:
        return None
    return mol


def load_molecules(path, limit=None):
    """Load from a .sdf, a directory of .sdf, or a directory of .xyz."""
    mols = []

    if osp.isfile(path) and path.endswith(".sdf"):
        for m in Chem.SDMolSupplier(path, removeHs=False, sanitize=True):
            if m is not None:
                mols.append(m)
            if limit and len(mols) >= limit:
                break
        return mols

    if osp.isdir(path):
        sdfs = sorted(f for f in os.listdir(path) if f.endswith(".sdf"))
        for f in sdfs:
            for m in Chem.SDMolSupplier(osp.join(path, f),
                                        removeHs=False, sanitize=True):
                if m is not None:
                    mols.append(m)
                if limit and len(mols) >= limit:
                    return mols
        if mols:
            return mols

        xyzs = sorted(f for f in os.listdir(path) if f.endswith(".xyz"))
        for f in xyzs:
            m = mol_from_xyz(osp.join(path, f))
            if m is not None:
                mols.append(m)
            if limit and len(mols) >= limit:
                break
        return mols

    raise SystemExit(f"cannot read molecules from {path}")


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------

def group_counts(mols):
    """
    For each functional group: the fraction of molecules containing it, and
    the mean number of occurrences per molecule. Paper 1's cyano finding is a
    statement about the second quantity.
    """
    n = len(mols)
    present = Counter()
    total = Counter()
    for m in mols:
        for name, patt in PATTERNS.items():
            hits = m.GetSubstructMatches(patt)
            if hits:
                present[name] += 1
                total[name] += len(hits)
    return {
        name: {
            "frac_molecules": present[name] / n if n else 0.0,
            "mean_per_molecule": total[name] / n if n else 0.0,
            "total": total[name],
        }
        for name in PATTERNS
    }


def basic_properties(mols, sa_fn):
    rows = []
    for m in mols:
        row = {
            "n_atoms": m.GetNumAtoms(),
            "n_heavy": m.GetNumHeavyAtoms(),
            "mw": Descriptors.MolWt(m),
            "n_rings": Descriptors.RingCount(m),
            "n_aromatic_rings": Descriptors.NumAromaticRings(m),
            "tpsa": Descriptors.TPSA(m),
        }
        if sa_fn is not None:
            try:
                row["sa_score"] = sa_fn(m)
            except Exception:
                row["sa_score"] = float("nan")
        rows.append(row)
    return rows


def fingerprints(mols, n_bits=1024, radius=2):
    gen = AllChem.GetMorganGenerator(radius=radius, fpSize=n_bits)
    X = np.zeros((len(mols), n_bits), dtype=np.float32)
    for i, m in enumerate(mols):
        fp = gen.GetFingerprintAsNumPy(m)
        X[i] = fp
    return X


def pca_and_cluster(X, labels, n_components=5, n_clusters=8, seed=0):
    from sklearn.cluster import AgglomerativeClustering, Birch
    from sklearn.decomposition import PCA

    n_components = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    Z = pca.fit_transform(X)

    birch = Birch(n_clusters=None, threshold=0.5)
    birch.fit(Z)
    centroids = birch.subcluster_centers_

    k = min(n_clusters, len(centroids))
    if k >= 2:
        agg = AgglomerativeClustering(n_clusters=k)
        sub_labels = agg.fit_predict(centroids)
        assign = sub_labels[birch.predict(Z)]
    else:
        assign = np.zeros(len(Z), dtype=int)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": float(pca.explained_variance_ratio_.sum()),
        "coords": Z[:, :2].tolist(),
        "cluster": assign.tolist(),
        "set_label": labels,
    }


def compare(ref_stats, gen_stats, name):
    """Enrichment of each group in a generated set relative to the reference."""
    rows = []
    for g in PATTERNS:
        r = ref_stats[g]["mean_per_molecule"]
        v = gen_stats[g]["mean_per_molecule"]
        if r == 0 and v == 0:
            continue
        ratio = (v / r) if r > 0 else float("inf")
        rows.append({
            "group": g,
            "reference_per_mol": round(r, 4),
            f"{name}_per_mol": round(v, 4),
            "enrichment": round(ratio, 3) if np.isfinite(ratio) else "inf",
        })
    rows.sort(key=lambda d: (-d["enrichment"]
                             if isinstance(d["enrichment"], float) else -1e9))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", required=True,
                   help="OE62 training SDF, the baseline distribution")
    p.add_argument("--generated", action="append", required=True,
                   help="Directory of .xyz/.sdf, or a .sdf. Repeatable.")
    p.add_argument("--names", default=None,
                   help="Comma-separated labels for the --generated sets")
    p.add_argument("--out-dir", default="analysis")
    p.add_argument("--ref-limit", type=int, default=10000,
                   help="Subsample the reference; 10k is plenty for statistics")
    p.add_argument("--no-cluster", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    sa_fn = sa_scorer()
    if sa_fn is None:
        print("note: SA_Score unavailable, skipping synthetic complexity")

    names = (args.names.split(",") if args.names
             else [f"set{i}" for i in range(len(args.generated))])
    if len(names) != len(args.generated):
        raise SystemExit("--names must match the number of --generated")

    print(f"reading reference: {args.reference}")
    ref = load_molecules(args.reference, limit=args.ref_limit)
    print(f"  {len(ref)} molecules")
    ref_groups = group_counts(ref)
    ref_props = basic_properties(ref, sa_fn)

    results = {
        "reference": {
            "path": args.reference,
            "n_molecules": len(ref),
            "groups": ref_groups,
            "mean_properties": {
                k: float(np.nanmean([r[k] for r in ref_props]))
                for k in ref_props[0]
            },
        },
        "generated": {},
    }

    all_X, all_labels = [], []
    if not args.no_cluster:
        all_X.append(fingerprints(ref))
        all_labels += ["reference"] * len(ref)

    for path, name in zip(args.generated, names):
        print(f"\nreading {name}: {path}")
        mols = load_molecules(path)
        print(f"  {len(mols)} molecules recovered")
        if not mols:
            print("  nothing readable, skipping")
            continue

        groups = group_counts(mols)
        props = basic_properties(mols, sa_fn)

        results["generated"][name] = {
            "path": path,
            "n_molecules": len(mols),
            "groups": groups,
            "mean_properties": {
                k: float(np.nanmean([r[k] for r in props]))
                for k in props[0]
            },
            "enrichment_vs_reference": compare(ref_groups, groups, name),
        }

        if not args.no_cluster:
            all_X.append(fingerprints(mols))
            all_labels += [name] * len(mols)

        print(f"  top enrichments vs OE62:")
        for row in results["generated"][name]["enrichment_vs_reference"][:6]:
            print(f"    {row['group']:<18} {row['reference_per_mol']:>7} -> "
                  f"{row[f'{name}_per_mol']:>7}   x{row['enrichment']}")

    if not args.no_cluster and len(all_X) > 1:
        print("\nPCA + clustering ...")
        X = np.vstack(all_X)
        results["pca_clustering"] = pca_and_cluster(X, all_labels)
        print(f"  {len(results['pca_clustering']['explained_variance_ratio'])} "
              f"components explain "
              f"{results['pca_clustering']['cumulative_variance']:.1%}")

    out = osp.join(args.out_dir, "analysis.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")

    # A compact table for the paper, cyano first since that is Paper 1's claim
    print("\n" + "=" * 72)
    print("C#N (nitrile) — Paper 1 found this enriched in low-gap molecules")
    print("=" * 72)
    print(f"  reference  {ref_groups['nitrile_CN']['mean_per_molecule']:.4f} "
          f"per molecule, present in "
          f"{ref_groups['nitrile_CN']['frac_molecules']:.1%}")
    for name, d in results["generated"].items():
        g = d["groups"]["nitrile_CN"]
        print(f"  {name:<10} {g['mean_per_molecule']:.4f} per molecule, "
              f"present in {g['frac_molecules']:.1%}")


if __name__ == "__main__":
    main()
