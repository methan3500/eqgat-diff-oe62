# Swapping G-SchNet for equivariant diffusion in a property-driven molecular design loop

Code for replacing the generator in the closed-loop molecular design workflow of
Westermayr et al. with [EQGAT-diff](https://github.com/tuanle618/eqgat-diff), an
E(3)-equivariant diffusion model, while keeping the OE62 dataset and the
SchNet-style property surrogate fixed.

The loop is: **generate → perceive bonds → score with a surrogate → select the
extreme tail → fine-tune the generator → repeat.** Only the generator differs
from the original study, so the generator is the only variable.

---

## Why this might be useful to you

Most of what is here is not the loop itself — that part is short — but the
machinery needed to make a bond-aware diffusion generator work on a dataset
that stores no bonds, at a scale that fits free cloud GPUs. If you are
attempting something similar, the measurements below will save you time whether
or not you use this code.

---

## Results so far

All numbers come from a single run of 3,000 generated molecules on one NVIDIA
T4. Intervals are 95% Wilson bounds.

### Unconditional generation, 100-epoch model

| Metric | Value | 95% CI |
|---|---|---|
| validity | 23.9% | [22.4, 25.5] |
| molecule stability | 51.9% | [50.1, 53.7] |
| atom stability | 97.9% | [97.8, 98.0] |
| novelty | 0.996 | |
| uniqueness | 1.000 | |
| AnglesW1 ↓ | 6.65 | |
| BondLengthsW1 ↓ | 0.020 | |
| connected components | 1.29 | |

Of 3,000 samples, 718 pass validity. Among the reported failures, atom valence
(781) and kekulisation (671) dominate against 93 of other kinds — both
combinatorial in the size of the element alphabet.

### One biasing iteration

| Stage | Count | Share | Limited by |
|---|---|---|---|
| Generated | 3,000 | 100% | — |
| Written to disk | 1,557 | 51.9% | molecule stability |
| Single connected component | 261 | 8.7% | **bond perception** |
| Surviving agreement filter | 241 | 8.0% | surrogate disagreement |
| Selected (μ − 0.5σ) | 74 | 2.5% | selection threshold |

**The dominant loss is not generation.** Of the 718 molecules the generator's
own validity metric accepts, re-perceiving bonds from written coordinates
recovers 261 — about one third. The diffusion model predicts bond orders
natively; routing its output through a coordinate-only file format discards
them, and they cannot reliably be recovered.

### Property surrogates

Two SchNet-style regressors (455,809 parameters) on 51,925 OE62 molecules:
**0.2262 eV** and **0.2330 eV** validation MAE, against ~0.25 eV reported for
the original study's surrogate and 0.82 eV for a mean-predictor baseline. Both
overfit substantially — final training error is about an order of magnitude
below validation error — so these figures are a *lower bound* on uncertainty
for off-equilibrium generated structures.

### Sampling cost

**1.64 s per molecule** on a T4 (3,000 molecules in 82 minutes). Applied to the
original protocol's volume:

| | Molecules | GPU-hours |
|---|---|---|
| One iteration (their lower bound) | 40,000 | 18.2 |
| One iteration (their upper bound) | 90,000 | 41.0 |
| Full protocol | 280,000–990,000 | 128–451 |
| This work | 3,000 | 1.4 |

A single iteration at the original volume does not fit in a 12-hour session.
This is the direct cost of the substitution: autoregressive generation takes a
few dozen cheap forward passes per molecule, diffusion takes 500 denoising
steps over a fully connected graph.

---

## Status

| Component | State |
|---|---|
| OE62 → bonded SDF conversion | working |
| EQGAT-diff training on OE62 | working, 100 epochs |
| Property surrogates | working, two seeds |
| Screening and selection | working |
| Two-surrogate agreement filter | working |
| Single biasing iteration | working |
| **Multi-iteration trajectory** | **not yet run** |
| Cyano knock-out study | supported, not run |

---

## Layout

```
scripts/      the eight scripts, described below
configs/      oe62_full70.yaml -- the generator config these results used
notebooks/    cell-by-cell transcript of the Kaggle run, and its own README
paper/        paper_lncs.tex, the LNCS manuscript
```

---

## Requirements

- The [EQGAT-diff](https://github.com/tuanle618/eqgat-diff) repository and its
  conda environment (`environment.yml` in that repo). Everything here runs
  inside it.
- RDKit, ASE, PyTorch Geometric, scikit-learn — all present in that environment
  except scikit-learn, which `analyze_generated.py` needs.
- **OE62** is not included: it is 129 MB, above GitHub's file limit. Download
  `OE62.db` from the dataset release accompanying Stuke et al. (2020),
  *Atomic structures and orbital energies of 61,489 crystal-forming organic
  molecules*, Scientific Data 7, 58.

### Patching EQGAT-diff

Five edits register OE62 as a dataset in the upstream repository. Paths are
relative to the clone root, whose package directory is `eqgat_diff/`:

| File | Change |
|---|---|
| `eqgat_diff/experiments/hparams.py` | add `"oe62"` to the dataset choices |
| `eqgat_diff/experiments/run_train.py` | branch importing `OE62DataModule` |
| `eqgat_diff/experiments/run_evaluation.py` | the same branch |
| `eqgat_diff/experiments/data/data_info.py` | `full_atom_encoder_oe62`, selected when `dataset == "oe62"` |
| `eqgat_diff/experiments/sampling/utils.py` | `from scipy import histogram` → `from numpy import histogram` (removed from SciPy in 1.3) |

Then place `scripts/oe62_dataset.py` at
`eqgat_diff/experiments/data/oe62/oe62_dataset.py`, with an empty
`__init__.py` beside it. `notebooks/kaggle_commit_b_cells.py` applies all of
this automatically.

---

## Pipeline

```bash
# 0. What labels does the database actually carry?
python scripts/inspect_oe62.py --db OE62.db --n 3

# 1. Optional: how many molecules survive a given filter, without
#    paying for bond perception (~2 minutes)
python scripts/census_oe62.py --db OE62.db

# 2. Convert to bonded SDFs. Resumable: rerun to continue.
python scripts/prepare_oe62_sharded.py \
    --db OE62.db --shard-dir data/oe62/shards \
    --max-atoms 70 --shard-size 5000
python scripts/prepare_oe62_sharded.py \
    --merge --shard-dir data/oe62/shards --out-dir data/oe62/raw

# 3. Train the generator. Upstream script, our config.
#    ~7 h on a T4 for 100 epochs (4 min 08 s/epoch). Edit the paths in
#    the config first.
python experiments/run_train.py --conf configs/oe62_full70.yaml

# 4. Train two surrogates with different seeds (~2 h each)
python scripts/train_oracle.py --db OE62.db \
    --label-key hlgap --negate --target gap \
    --max-atoms 70 --limit 60000 --epochs 120 --seed 1 --out-dir oracle
python scripts/train_oracle.py --db OE62.db \
    --label-key hlgap --negate --target gap \
    --max-atoms 70 --limit 60000 --epochs 120 --seed 2 --out-dir oracle2

# 5. The biasing loop
python scripts/run_biasing_loop.py \
    --repo /path/to/eqgat-diff --dataset-root data/oe62 \
    --start-ckpt runs/oe62/run0/last.ckpt \
    --oracle oracle/oracle_gap.pt --oracle2 oracle2/oracle_gap.pt \
    --work-dir biasing --target gap --direction min \
    --n-loops 5 --n-sample 3000 --sigma-x 0.5 \
    --finetune-epochs 30 --warm-start-frac 0.1

# 6. Analysis (no GPU)
python scripts/analyze_generated.py \
    --reference data/oe62/raw/oe62_train.sdf \
    --generated biasing/loop0/samples --generated biasing/loop4/samples \
    --names loop0,loop4 --out-dir analysis

python scripts/ablate_perception.py \
    --samples biasing/loop0/samples --names loop0
```

None of this fits in one 12-hour session. `notebooks/README.md` gives the
per-stage timings and how the work was split across Kaggle commits, with the
two Kaggle-specific traps that cost us runs.

---

## What each script does

| Script | Purpose |
|---|---|
| `inspect_oe62.py` | Print what labels the ASE database carries, and under what names. Run first. |
| `census_oe62.py` | Count filter yields without bond perception, so you can pick a size/element cut from real numbers in two minutes. |
| `prepare_oe62_sharded.py` | OE62 → bonded SDFs. Parallel, resumable, with a per-molecule timeout. `--exclude-smarts` supports knock-out studies. |
| `oe62_dataset.py` | PyTorch Geometric dataset and datamodule for EQGAT-diff. Goes inside the upstream repo. |
| `train_oracle.py` | SchNet-style property surrogate on coordinates and atomic numbers. |
| `run_biasing_loop.py` | The loop: generate, perceive, score, agree, select, fine-tune, repeat. |
| `analyze_generated.py` | Functional-group enrichment, PCA, clustering, synthetic accessibility. |
| `ablate_perception.py` | Compares three bond-perception strategies on the same structures. CPU only. |

---

## Things that cost us time

Recorded because none of them are obvious, and each one cost hours.

**`rdDetermineBonds` can hang forever.** It runs a combinatorial bond-order
search with no internal time limit. Median perception is under 10 ms, but a few
molecules never finish — one stalled a conversion for over an hour with no
error and no output. The runtime distribution is bimodal, so a short timeout
costs almost nothing. `signal.alarm` does not help: the search is a C++ call and
Python defers signal handlers until it returns, so perception has to run in a
killable child process. Measured: 55 of 61,489 molecules hit a 5-second timeout,
all polynitro or polysulfur species.

**Bond perception silently removes whole elements.** It fails on 3,067
molecules, and that failure is not uniform across the periodic table: it removes
*every* molecule containing Se, As, Te or Li. The generator therefore sees 12 of
OE62's 16 elements. Check element coverage after conversion, not before.

**Forcing neutral charge during perception throws away most generated
molecules.** Generated structures include ions. Trying net charges
{0, ±1, ±2} roughly doubles the yield. `ablate_perception.py` measures this.

**Scoring and fine-tuning need different things.** The surrogate reads atomic
numbers and coordinates, so connectivity alone is enough to score a molecule.
Full bond orders are needed only to write an SDF for fine-tuning. Treating these
as one requirement discards molecules unnecessarily.

**Fine-tuning on a small selection breaks checkpoint loading.** The model
registers `empirical_num_nodes` sized by the largest molecule in its training
set. A selection whose biggest molecule is smaller than the original's produces
a smaller buffer and fails with `size mismatch for empirical_num_nodes`.
`oe62_dataset.py` pads the node-count histogram to a fixed range so every
subset produces the same buffer size.

**`run_evaluation.py` needs `train_dataset` on the datamodule.** It builds the
datamodule with `only_stats=True`, a path that never reaches
`super().__init__()`, and then reads `datamodule.train_dataset.smiles`. The
attribute has to be assigned explicitly.

**One molecule in 43,656 breaks SMILES canonicalisation.** It sanitises fine,
then raises `Pre-condition Violation: bad bond type` in `MolToSmiles`, killing a
41-minute cache build. Caught per-molecule rather than per-run.

---

## Known deviations from the original study

Stated plainly, because they bound what the results can claim.

- **Property.** We regress the HOMO–LUMO gap stored in OE62, a DFT quantity.
  The original targets *GW* fundamental gaps via Δ-learning. Absolute values are
  not comparable; only displacements are.
- **Elements.** 12 of 16, for the reason above. The original study's finding
  that low-gap molecules are enriched in **selenium** therefore cannot be
  tested here at all.
- **Scale.** ~3,000 molecules per iteration against their 40,000–90,000.
- **First iteration.** They seed the loop by fine-tuning on the extreme decile
  of OE62 itself; `--warm-start-frac 0.1` reproduces this, but scores the
  training set with the surrogate rather than using stored labels.
- **Agreement threshold.** They discard when two surrogates differ by more than
  the *average* of their MAEs; our default is the *sum*, which is looser.
- **Descriptors.** Morgan fingerprints with PCA in place of SOAP, and RDKit's
  SA_Score in place of SCScore. Five components explain 12% of variance here
  against >98% for their SOAP descriptors, so our embedding is not a faithful
  low-dimensional map.
- **No G₀W₀ verification.** Property values are surrogate predictions.

---

## Citation

If this code is useful, please also cite the two papers it builds on:

```bibtex
@article{westermayr2022,
  title  = {High-throughput property-driven generative design of
            functional organic molecules},
  author = {Westermayr, Julia and Gilkes, Joe and Barrett, Rhyan and
            Maurer, Reinhard J.},
  journal = {arXiv preprint arXiv:2207.01476},
  year   = {2022}
}

@article{le2023,
  title  = {Navigating the design space of equivariant diffusion-based
            generative models for de novo 3D molecule generation},
  author = {Le, Tuan and Cremer, Julian and No{\'e}, Frank and
            Clevert, Djork-Arn{\'e} and Sch{\"u}tt, Kristof T.},
  journal = {arXiv preprint arXiv:2309.17296},
  year   = {2023}
}
```

---

## License and attribution

The code here is MIT licensed — see `LICENSE`.

This repository does not redistribute EQGAT-diff or OE62; it contains only
scripts that drive them, plus the patches needed to register OE62 as a dataset.
`scripts/oe62_dataset.py` is structured after EQGAT-diff's own
`qm9_dataset.py` and subclasses its `AbstractAdaptiveDataModule`, so it is an
adaptation of that project's code rather than an independent implementation.
At the time of writing the EQGAT-diff repository carries no LICENSE file;
anyone building on it should check its current terms, and credit belongs to
its authors.

OE62 is distributed by Stuke et al. through Materials Cloud under its own
terms.
