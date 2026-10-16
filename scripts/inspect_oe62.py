"""
inspect_oe62.py

RUN THIS FIRST. It answers the one question that decides whether an oracle
can be trained at all: does OE62.db carry electronic-structure labels, and
under what names?

An ASE database row can hold extra values in two places:
  row.key_value_pairs  -> scalars (floats, ints, strings)
  row.data             -> arbitrary arrays (e.g. full eigenvalue spectra)

Paper 1's SchNet+H predicts whole eigenvalue spectra and reads HOMO/LUMO as
columns 50 and 51 of the sorted array, so if the labels are here at all they
are most likely a per-molecule array in row.data.

USAGE
-----
    conda run -n eqgatdiff python inspect_oe62.py --db /content/data/train.db
"""

import argparse

import numpy as np
from ase.db import connect


def describe(value, indent="      "):
    """Print a short, safe description of an arbitrary stored value."""
    if isinstance(value, np.ndarray):
        flat = value.ravel()
        preview = np.array2string(flat[:6], precision=4, separator=", ")
        return (f"ndarray shape={value.shape} dtype={value.dtype}\n"
                f"{indent}first values: {preview}")
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__} len={len(value)} first={value[:5]}"
    if isinstance(value, (int, float, np.floating, np.integer)):
        return f"{type(value).__name__} = {value}"
    if isinstance(value, str):
        return f"str = {value[:80]!r}"
    return f"{type(value).__name__}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--n", type=int, default=3,
                        help="How many rows to inspect")
    args = parser.parse_args()

    db = connect(args.db)
    total = db.count()
    print(f"database : {args.db}")
    print(f"rows     : {total}\n")

    for i in range(1, args.n + 1):
        try:
            row = db.get(i)
        except Exception as exc:
            print(f"row {i}: could not read ({exc})")
            continue

        atoms = row.toatoms()
        print(f"--- row {i} ---")
        print(f"  formula    : {atoms.get_chemical_formula()}")
        print(f"  n_atoms    : {len(atoms)}")

        kvp = dict(row.key_value_pairs) if row.key_value_pairs else {}
        print(f"  key_value_pairs ({len(kvp)}):")
        if not kvp:
            print("      (none)")
        for k, v in kvp.items():
            print(f"      {k}: {describe(v)}")

        try:
            data = dict(row.data) if row.data else {}
        except Exception:
            data = {}
        print(f"  data ({len(data)}):")
        if not data:
            print("      (none)")
        for k, v in data.items():
            print(f"      {k}: {describe(v)}")
        print()

    print("=" * 66)
    print("HOW TO READ THIS")
    print("=" * 66)
    print("""
Look for anything resembling electronic-structure output. Likely names:
  eigenvalues, eigenvalues_pbe0, energies, homo, lumo, gap,
  delta_eigenvalues_pbe0_gbw, ea, ip

THREE POSSIBLE OUTCOMES

1. An eigenvalue ARRAY is present (e.g. shape (100,) or similar).
   Best case. HOMO/LUMO are two adjacent entries of the sorted spectrum
   (Paper 1 uses indices 50 and 51 for OE62). An oracle can be trained.

2. Only SCALARS like homo / lumo / gap are present.
   Also fine, and simpler: train the oracle to regress those directly.

3. key_value_pairs and data are both empty.
   The file is geometry-only. No oracle can be trained from it. You would
   need the labelled OE62 release from Materials Cloud (Stuke et al. 2020),
   which ships the G0W0@PBE0 values, and re-do this check on that file.

Report which case you are in before going further.
""")


if __name__ == "__main__":
    main()
