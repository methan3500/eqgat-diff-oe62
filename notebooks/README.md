# Running this on Kaggle

`kaggle_commit_b_cells.py` is **not a runnable script** — it is a cell-by-cell
transcript of a Kaggle notebook, including `%%time` magics, meant to be pasted
one block per cell. It will raise `SyntaxError` if you run it with `python`.

## Why it is split into commits

A Kaggle session is capped at 12 hours and the weekly GPU budget is 30 hours.
The full pipeline does not fit in one session:

| Stage | Time on a T4 |
|---|---|
| Environment + repo + upstream patches | ~25 min |
| OE62 → bonded SDFs (43,657 molecules) | ~45 min |
| Dataset cache build | ~41 min |
| Generator, 100 epochs | ~7 h 05 m (4 min 08 s/epoch) |
| Surrogate, 60,000 molecules, 120 epochs | ~2 h |
| Sampling, 3,000 molecules | ~1 h 22 m (1.64 s/molecule) |

So the work is split across commits, each under 12 hours, with the previous
commit's `/kaggle/working` attached as an input dataset to the next. **Save
Version → Save & Run All (Commit)** is what persists output; an interactive
session does not.

Commit B, transcribed here, assumes two attached inputs: the scripts and
`OE62.db`, and the previous commit's output (environment tarball, built
`raw/` and `processed/`, the generator checkpoint, and the first surrogate).
It retrains the second surrogate from scratch, because an earlier attempt
trained it on only 10,000 molecules — pass `--limit 60000` explicitly, since
the script's default is 10,000.

## Two traps specific to Kaggle

**Stale script copies.** `/kaggle/input` may hold several versions of the same
file across attached datasets. Cell 1 ranks candidates — a dataset mount over a
working-directory copy, then shallower paths, then newer modification times —
rather than taking the first glob hit.

**Copying a directory that does not exist yet.** `cp -r SRC/raw DEST/` flattens
into `DEST` when `DEST` is missing, instead of creating `DEST/raw`. Every
destination is created with `os.makedirs(..., exist_ok=True)` first, and the
cell asserts on file sizes afterwards.
