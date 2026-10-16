"""
census_oe62.py

Counts how many OE62 molecules survive a given (element whitelist, size cap)
filter, WITHOUT doing bond perception. Perception is the slow part of
prepare_oe62_filtered.py; counting is not. This lets you choose a filter tier
from real numbers in about two minutes instead of guessing and then discovering
the yield 40 minutes into a conversion.

The counts it prints are upper bounds on the final dataset size: roughly 3% of
survivors are additionally lost to bond-perception failure or fragmentation
(measured: 218 lost out of 6,814 on the CHNO / 30-atom tier).

USAGE (inside the eqgatdiff conda environment)
-----------------------------------------------
    conda run --no-capture-output -n eqgatdiff python census_oe62.py \
        --db /content/data/train.db

Add --tiers to test your own combinations:

    ... --tiers "C,H,N,O:30  C,H,N,O:50  C,H,N,O,F,S:40  ALL:50"

where ALL means every OE62 element, and the number after the colon is the
maximum atom count.
"""

import argparse
from collections import Counter

from ase.io import read as ase_read

OE62_ELEMENTS = {
    "H", "Li", "B", "C", "N", "O", "F", "Si",
    "P", "S", "Cl", "As", "Se", "Br", "Te", "I",
}

DEFAULT_TIERS = [
    ("C,H,N,O", 30),
    ("C,H,N,O", 40),
    ("C,H,N,O", 50),
    ("C,H,N,O,F", 40),
    ("C,H,N,O,F,S", 40),
    ("C,H,N,O,F,S,Cl", 50),
    ("ALL", 50),
    ("ALL", 70),
    ("ALL", 999),
]

# Empirical loss to bond perception + fragmentation, from the measured
# CHNO/30 run: 218 failures out of 6,814 that passed the size/element gates.
PERCEPTION_SURVIVAL = 0.968


def parse_tiers(spec):
    tiers = []
    for token in spec.split():
        elements, _, cap = token.rpartition(":")
        tiers.append((elements, int(cap)))
    return tiers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True, help="Path to OE62 ASE .db file")
    parser.add_argument("--tiers", default=None,
                        help='Space-separated "ELEMENTS:MAXATOMS" tiers. '
                             'ELEMENTS may be ALL.')
    args = parser.parse_args()

    tiers = parse_tiers(args.tiers) if args.tiers else DEFAULT_TIERS

    print(f"Reading {args.db} ...")
    geoms = ase_read(args.db, ":")
    n_total = len(geoms)
    print(f"  loaded {n_total} molecules\n")

    # Cache one (size, element-set) pair per molecule so each tier is a cheap
    # pass over tuples rather than over ASE objects.
    profiles = []
    size_hist = Counter()
    element_hist = Counter()
    for atoms in geoms:
        symbols = frozenset(atoms.get_chemical_symbols())
        n = len(atoms)
        profiles.append((n, symbols))
        size_hist[min(n // 10 * 10, 150)] += 1
        for s in symbols:
            element_hist[s] += 1

    print("Size distribution (atoms per molecule, 10-atom bins)")
    cumulative = 0
    for lo in sorted(size_hist):
        cumulative += size_hist[lo]
        label = f"{lo}-{lo + 9}" if lo < 150 else "150+"
        print(f"  {label:>8} : {size_hist[lo]:>7}   "
              f"cumulative {cumulative:>7} ({100 * cumulative / n_total:5.1f}%)")

    print("\nElement prevalence (molecules containing the element)")
    for sym, count in element_hist.most_common():
        print(f"  {sym:>3} : {count:>7} ({100 * count / n_total:5.1f}%)")

    print("\nTier yields")
    print(f"  {'elements':<18} {'max atoms':>9} {'passes gates':>13} "
          f"{'est. usable':>12} {'% of OE62':>10}")
    for elements, cap in tiers:
        allowed = OE62_ELEMENTS if elements.upper() == "ALL" else {
            e.strip() for e in elements.split(",") if e.strip()
        }
        unknown = allowed - OE62_ELEMENTS
        if unknown:
            raise SystemExit(f"Not OE62 elements: {sorted(unknown)}")

        passing = sum(
            1 for n, symbols in profiles
            if n <= cap and symbols.issubset(allowed)
        )
        usable = int(passing * PERCEPTION_SURVIVAL)
        print(f"  {elements:<18} {cap:>9} {passing:>13} {usable:>12} "
              f"{100 * passing / n_total:>9.1f}%")

    print("\nThe 'est. usable' column applies a 96.8% bond-perception survival "
          "rate measured on the CHNO/30 tier. Treat it as approximate.")


if __name__ == "__main__":
    main()
