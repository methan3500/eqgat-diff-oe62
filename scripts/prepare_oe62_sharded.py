"""
prepare_oe62_sharded.py  (v2)

Converts the OE62 ASE database into SDF files with bonds perceived by RDKit,
in resumable shards, with a per-molecule timeout.

WHY THE TIMEOUT EXISTS
----------------------
rdDetermineBonds runs a combinatorial bond-order search with no internal time
limit. On most OE62 molecules it finishes in under 10 ms, but on a few — large
conjugated or charged systems — it can run effectively forever. A single such
molecule stalls an entire conversion with no error and no progress output.
Measured: shards 0-2 (15,000 molecules) took about five minutes; shard 3 hung
for over an hour on one structure.

A signal.alarm timeout does not help here, because the search is a C++ call and
Python defers signal handlers until it returns. So perception runs in a child
process that the parent can terminate. The child is persistent and is only
respawned when it has to be killed, so the per-molecule cost stays low.

NOTE ON ORDERING
----------------
OE62 appears to be sorted by molecule size: shards 0-2 contained only 46
molecules above 70 atoms, against a database-wide figure of 6.7%. Later shards
will therefore be slower and keep fewer molecules. This is expected.

USAGE
-----
Convert (rerun the same command to resume; finished shards are skipped):

    conda run --no-capture-output -n eqgatdiff python prepare_oe62_sharded.py \\
        --db /content/data/train.db \\
        --shard-dir /content/eqgat-diff/data/oe62/shards \\
        --max-atoms 70 --shard-size 5000 --timeout 20

Then merge into train/val/test:

    conda run --no-capture-output -n eqgatdiff python prepare_oe62_sharded.py \\
        --merge \\
        --shard-dir /content/eqgat-diff/data/oe62/shards \\
        --out-dir /content/eqgat-diff/data/oe62/raw
"""

import argparse
import glob
import multiprocessing as mp
import os
import os.path as osp
import time

import numpy as np
from ase.db import connect
from rdkit import Chem, RDLogger
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

RDLogger.DisableLog("rdApp.*")


def _matches(molblock, patt):
    """True if the molblock contains the SMARTS pattern."""
    try:
        m = Chem.MolFromMolBlock(molblock, removeHs=False, sanitize=True)
        return m is not None and m.HasSubstructMatch(patt)
    except Exception:
        return False


OE62_ELEMENTS = {
    "H", "Li", "B", "C", "N", "O", "F", "Si",
    "P", "S", "Cl", "As", "Se", "Br", "Te", "I",
}


# --------------------------------------------------------------------------
# Bond perception, run inside a killable child process
# --------------------------------------------------------------------------

def perceive(symbols, positions, charge=0):
    """
    Build an RDKit mol from atoms + coordinates and perceive bonds.
    Returns a molblock string, or None if perception failed or the result
    was fragmented.
    """
    rw = Chem.RWMol()
    for sym in symbols:
        rw.AddAtom(Chem.Atom(sym))

    conf = Chem.Conformer(rw.GetNumAtoms())
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, Point3D(float(pos[0]), float(pos[1]), float(pos[2])))

    mol = rw.GetMol()
    mol.AddConformer(conf, assignId=True)

    try:
        rdDetermineBonds.DetermineBonds(mol, charge=charge)
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    # EQGAT-diff, like G-SchNet, expects single connected molecules.
    if len(Chem.GetMolFrags(mol)) != 1:
        return None

    return Chem.MolToMolBlock(mol, includeStereo=True)


def _worker(in_q, out_q):
    """Persistent child: read jobs, return molblocks."""
    while True:
        job = in_q.get()
        if job is None:
            return
        symbols, positions = job
        try:
            out_q.put(perceive(symbols, positions))
        except Exception:
            out_q.put(None)


class PerceptionPool:
    """
    A pool of perception workers, each in its own process, with a per-molecule
    timeout.

    Perception is single-threaded and CPU-bound, so running one worker per core
    is close to a linear speed-up. The timeout matters more than it looks: the
    runtime distribution is bimodal — almost every molecule finishes in under
    10 ms, and a few blow up combinatorially and never finish. Those few are
    what made the later shards slow, because each one held a worker for the
    whole timeout.

    A worker that exceeds the timeout is killed and replaced, so one bad
    molecule costs `timeout` seconds on one core rather than stalling the run.
    """

    def __init__(self, n_workers, timeout):
        self.n = max(1, n_workers)
        self.timeout = timeout
        self.ctx = mp.get_context("fork")
        self.procs = [None] * self.n
        self.in_q = [None] * self.n
        self.out_q = [None] * self.n
        # busy[i] is None, or (payload_id, start_time) for the in-flight job
        self.busy = [None] * self.n
        for i in range(self.n):
            self._start(i)

    def _start(self, i):
        self.in_q[i] = self.ctx.Queue()
        self.out_q[i] = self.ctx.Queue()
        self.procs[i] = self.ctx.Process(
            target=_worker, args=(self.in_q[i], self.out_q[i]), daemon=True
        )
        self.procs[i].start()
        self.busy[i] = None

    def _kill(self, i):
        try:
            self.procs[i].terminate()
            self.procs[i].join(timeout=5)
        except Exception:
            pass
        self._start(i)

    def submit(self, i, job_id, symbols, positions):
        self.in_q[i].put((symbols, positions))
        self.busy[i] = (job_id, time.time())

    def idle_slots(self):
        return [i for i in range(self.n) if self.busy[i] is None]

    def harvest(self):
        """
        Collect whatever has finished or timed out.
        Returns a list of (job_id, molblock_or_None, timed_out).
        """
        done = []
        for i in range(self.n):
            if self.busy[i] is None:
                continue
            job_id, started = self.busy[i]

            try:
                block = self.out_q[i].get_nowait()
                self.busy[i] = None
                done.append((job_id, block, False))
                continue
            except Exception:
                pass

            if time.time() - started > self.timeout:
                self._kill(i)
                done.append((job_id, None, True))
            elif not self.procs[i].is_alive():
                # Child died (segfault, OOM). Treat as a failure, not a timeout.
                self._kill(i)
                done.append((job_id, None, False))
        return done

    def in_flight(self):
        return sum(1 for b in self.busy if b is not None)

    def close(self):
        for i in range(self.n):
            try:
                self.in_q[i].put(None)
                self.procs[i].join(timeout=2)
            except Exception:
                pass
            if self.procs[i] is not None and self.procs[i].is_alive():
                self.procs[i].terminate()


# --------------------------------------------------------------------------
# Shard bookkeeping
# --------------------------------------------------------------------------

def shard_path(shard_dir, index):
    return osp.join(shard_dir, f"shard_{index:04d}.sdf")


def done_path(shard_dir, index):
    return osp.join(shard_dir, f"shard_{index:04d}.done")


def convert(args):
    os.makedirs(args.shard_dir, exist_ok=True)

    if args.elements:
        allowed = {e.strip() for e in args.elements.split(",") if e.strip()}
        unknown = allowed - OE62_ELEMENTS
        if unknown:
            raise SystemExit(f"Not OE62 elements: {sorted(unknown)}")
        print(f"element whitelist: {sorted(allowed)}", flush=True)
    else:
        allowed = OE62_ELEMENTS
        print("element whitelist: all 16 OE62 elements", flush=True)

    exclude_patt = None
    if args.exclude_smarts:
        exclude_patt = Chem.MolFromSmarts(args.exclude_smarts)
        if exclude_patt is None:
            raise SystemExit(f"bad SMARTS: {args.exclude_smarts}")
        print(f"knock-out: excluding molecules matching "
              f"{args.exclude_smarts}", flush=True)

    db = connect(args.db)
    total = db.count()
    n_shards = (total + args.shard_size - 1) // args.shard_size
    print(f"database: {total} rows -> {n_shards} shards of {args.shard_size}",
          flush=True)
    print(f"perception workers: {args.workers}  "
          f"(per-molecule timeout {args.timeout} s)\n", flush=True)

    pending = [i for i in range(n_shards)
               if not osp.exists(done_path(args.shard_dir, i))]
    if not pending:
        print("All shards already converted. Run again with --merge.", flush=True)
        return

    print(f"shards remaining: {len(pending)} of {n_shards}", flush=True)
    if args.max_shards:
        pending = pending[: args.max_shards]
        print(f"processing {len(pending)} this run (--max-shards)", flush=True)
    print(flush=True)

    pool = PerceptionPool(args.workers, args.timeout)

    try:
        for position, shard_index in enumerate(pending):
            lo = shard_index * args.shard_size
            hi = min(lo + args.shard_size, total)
            selection = f"id>={lo + 1},id<={hi}"

            t0 = time.time()
            n_too_big = n_bad_element = 0

            # Gather this shard's candidates first. Holding ~5,000 molecules of
            # symbols and coordinates is a few MB, and it keeps the database
            # read separate from the parallel perception below.
            candidates = []
            for row in db.select(selection):
                atoms = row.toatoms()
                if len(atoms) > args.max_atoms or len(atoms) < args.min_atoms:
                    n_too_big += 1
                    continue
                symbols = atoms.get_chemical_symbols()
                if not set(symbols).issubset(allowed):
                    n_bad_element += 1
                    continue
                candidates.append((row.id, len(atoms),
                                   atoms.get_chemical_formula(),
                                   symbols, atoms.get_positions()))

            print(f"  shard {shard_index:04d}: {len(candidates)} candidates "
                  f"({n_too_big} over {args.max_atoms} atoms, "
                  f"{n_bad_element} wrong element)", flush=True)

            blocks = []
            n_failed = n_timeout = n_excluded = 0
            next_job = 0
            completed = 0
            last_report = time.time()

            while completed < len(candidates):
                # Keep every worker fed.
                for slot in pool.idle_slots():
                    if next_job >= len(candidates):
                        break
                    _, _, _, symbols, positions = candidates[next_job]
                    pool.submit(slot, next_job, symbols, positions)
                    next_job += 1

                finished = pool.harvest()
                if not finished:
                    time.sleep(0.002)
                    continue

                for job_id, block, timed_out in finished:
                    completed += 1
                    if timed_out:
                        n_timeout += 1
                        row_id, n_atoms, formula, _, _ = candidates[job_id]
                        print(f"    timeout: row {row_id}, {n_atoms} atoms, "
                              f"{formula}", flush=True)
                    elif block is None:
                        n_failed += 1
                    elif exclude_patt is not None and _matches(block, exclude_patt):
                        # Knock-out mode: drop molecules containing the
                        # pattern, so the model never sees that chemistry.
                        # Must happen after perception, because the pattern is
                        # a bond motif and OE62 stores no bonds.
                        n_excluded += 1
                    else:
                        blocks.append(block)

                if time.time() - last_report > 30:
                    rate = completed / max(time.time() - t0, 1e-6)
                    print(f"    shard {shard_index:04d}: {completed}/"
                          f"{len(candidates)} ({rate:.0f}/s, kept "
                          f"{len(blocks)}, timeouts {n_timeout})", flush=True)
                    last_report = time.time()

            with open(shard_path(args.shard_dir, shard_index), "w") as fh:
                for block in blocks:
                    fh.write(block)
                    if not block.endswith("\n"):
                        fh.write("\n")
                    fh.write("$$$$\n")

            # The .done marker goes last, so an interrupted shard is redone
            # rather than silently counted as complete.
            with open(done_path(args.shard_dir, shard_index), "w") as fh:
                fh.write(
                    f"kept {len(blocks)}\ntoo_big {n_too_big}\n"
                    f"bad_element {n_bad_element}\nfailed {n_failed}\n"
                    f"timeout {n_timeout}\nexcluded {n_excluded}\n"
                )

            elapsed = time.time() - t0
            remaining = len(pending) - position - 1
            print(f"  shard {shard_index:04d} done: kept {len(blocks)}, "
                  f"failed {n_failed}, timed out {n_timeout}, "
                  f"knocked out {n_excluded}  "
                  f"({elapsed / 60:.1f} min)", flush=True)
            if remaining:
                print(f"  ~{remaining * elapsed / 60:.0f} min left at this rate\n",
                      flush=True)
    finally:
        pool.close()

    still = [i for i in range(n_shards)
             if not osp.exists(done_path(args.shard_dir, i))]
    if still:
        print(f"\n{len(still)} shards still pending. Rerun this cell to continue.",
              flush=True)
    else:
        print("\nAll shards done. Run again with --merge.", flush=True)


def merge(args):
    os.makedirs(args.out_dir, exist_ok=True)

    shards = sorted(glob.glob(osp.join(args.shard_dir, "shard_*.sdf")))
    if not shards:
        raise SystemExit(f"No shard files in {args.shard_dir}")
    print(f"merging {len(shards)} shards", flush=True)

    mols = []
    for path in shards:
        suppl = Chem.SDMolSupplier(path, removeHs=False, sanitize=True)
        before = len(mols)
        for mol in suppl:
            if mol is not None:
                mols.append(mol)
        print(f"  {osp.basename(path)}: {len(mols) - before}", flush=True)

    print(f"\ntotal molecules: {len(mols)}", flush=True)
    if len(mols) == 0:
        raise SystemExit("No molecules recovered from the shards.")

    sizes = np.array([m.GetNumAtoms() for m in mols])
    print(f"atoms per molecule: min {sizes.min()}, mean {sizes.mean():.1f}, "
          f"max {sizes.max()}", flush=True)

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(len(mols))
    n_train = int(args.train_frac * len(mols))
    n_val = int(args.val_frac * len(mols))

    splits = {
        "train": idx[:n_train],
        "val": idx[n_train:n_train + n_val],
        "test": idx[n_train + n_val:],
    }

    for split_name, split_idx in splits.items():
        path = osp.join(args.out_dir, f"oe62_{split_name}.sdf")
        writer = Chem.SDWriter(path)
        for i in split_idx:
            writer.write(mols[int(i)])
        writer.close()
        print(f"  wrote {len(split_idx):>6} molecules -> {path}", flush=True)

    print("\nDone. Delete <dataset_root>/processed before training so PyTorch "
          "Geometric rebuilds its cache from these files.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", help="Path to OE62 ASE .db file (convert mode)")
    p.add_argument("--shard-dir", required=True)
    p.add_argument("--out-dir", help="Where to write the split SDFs (merge mode)")

    p.add_argument("--merge", action="store_true",
                   help="Merge completed shards into train/val/test and exit")

    p.add_argument("--max-atoms", type=int, default=70)
    p.add_argument("--min-atoms", type=int, default=0)
    p.add_argument("--elements", default=None,
                   help="Comma-separated whitelist. Omit for all 16 elements.")
    p.add_argument("--shard-size", type=int, default=5000)
    p.add_argument("--max-shards", type=int, default=None,
                   help="Process at most this many shards this run")
    p.add_argument("--timeout", type=float, default=5.0,
                   help="Seconds allowed per molecule before giving up on it. "
                        "The runtime distribution is bimodal (under 10 ms, or "
                        "never), so a small value costs almost nothing.")
    p.add_argument("--exclude-smarts", default=None,
                   help="Drop molecules matching this SMARTS, for knock-out "
                        "studies. Paper 1's cyano knock-out is "
                        "'[CX2]#[NX1]'. Applied after bond perception.")
    p.add_argument("--workers", type=int, default=0,
                   help="Parallel perception processes. 0 = one per CPU core.")

    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.workers <= 0:
        args.workers = max(1, os.cpu_count() or 1)

    if args.merge:
        if not args.out_dir:
            raise SystemExit("--merge needs --out-dir")
        merge(args)
    else:
        if not args.db:
            raise SystemExit("convert mode needs --db")
        convert(args)


if __name__ == "__main__":
    main()
