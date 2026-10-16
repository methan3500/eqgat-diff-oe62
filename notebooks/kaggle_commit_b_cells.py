# ======================================================================
# COMMIT B - the finishing run.  Total ~10.2 h, inside the 12 h limit.
#
# Attach exactly TWO inputs:
#     oe62-eqgat            (OE62.db + the seven scripts)
#     notebookc78c17139b    (env tarball, full raw/ + processed/,
#                            generator last.ckpt, oracle 1)
# Commit A's output is NOT needed -- oracle 2 is retrained here in Cell 5,
# because the earlier one only saw 10,000 molecules.
#
# Settings: Accelerator = GPU T4 x2,  Internet = On
# Run with: Save Version -> Save & Run All (Commit)
#
# Paste each block below into its own cell, in order. Keep the leading "#"
# on comment lines, and keep %%time as the first line of its cell.
# ======================================================================


# ====================================================================
# CELL 1 - setup (conda, repo, env, scripts, upstream patches)
# ====================================================================

%%time
import os, glob, re, shutil, subprocess

CONDA = "/opt/miniconda3/bin/conda"
WORK  = "/kaggle/working"
REPO  = f"{WORK}/eqgat-diff"
PKG   = f"{REPO}/eqgat_diff/experiments/data/oe62"
DROOT = f"{REPO}/data/oe62"
BIN   = f"{WORK}/scripts"

# ---- find the inputs wherever Kaggle mounted them --------------------
db_hits = glob.glob('/kaggle/input/**/OE62.db', recursive=True)
assert db_hits, "OE62.db not found — attach your dataset to this notebook"
DB  = db_hits[0]
SRC = os.path.dirname(DB)
print("inputs at:", SRC)

# ---- miniconda -------------------------------------------------------
if not os.path.exists(CONDA):
    !wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    !bash /tmp/miniconda.sh -b -p /opt/miniconda3
!PYTHONPATH= {CONDA} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
!PYTHONPATH= {CONDA} tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# ---- repo: clone into an empty dir, or the clone silently fails ------
if not os.path.exists(f"{REPO}/eqgat_diff/environment.yml"):
    !rm -rf {REPO}
    !git clone -q https://github.com/tuanle618/eqgat-diff.git {REPO}
assert os.path.exists(f"{REPO}/eqgat_diff/experiments/hparams.py"), "clone failed — Internet off?"

# ---- conda env: unpack a saved tarball if present, else build --------
envs = subprocess.run([CONDA, "env", "list"], capture_output=True, text=True).stdout
tarball = glob.glob('/kaggle/input/**/eqgatdiff_env.tar.gz', recursive=True)

if "eqgatdiff" not in envs:
    if tarball:
        print("restoring env from", tarball[0], "(~2 min)")
        os.makedirs("/opt/miniconda3/envs", exist_ok=True)
        !tar -xzf {tarball[0]} -C /opt/miniconda3/envs
    else:
        print("building env from scratch (~20 min)")
        !PYTHONPATH= {CONDA} env create -q -f {REPO}/eqgat_diff/environment.yml
        !PYTHONPATH= {CONDA} run -n eqgatdiff pip install -q "setuptools==69.5.1" wheel

!sed -i 's/^from scipy import histogram/from numpy import histogram/' \
    {REPO}/eqgat_diff/experiments/sampling/utils.py

# ---- scripts: find them anywhere in the inputs, newest copy wins ------
os.makedirs(BIN, exist_ok=True)

NEEDED = ["oe62_dataset.py", "prepare_oe62_sharded.py", "train_oracle.py",
          "run_biasing_loop.py", "analyze_generated.py", "census_oe62.py",
          "inspect_oe62.py"]

# Several copies exist across the inputs. Rank: a dataset copy beats one
# baked into an old notebook output; a shallower path beats a nested one
# (new uploads land at the root); newer mtime beats older.
cands = {}
for p in glob.glob('/kaggle/input/**/*.py', recursive=True):
    name = re.sub(r'-\d+\.py$', '.py', os.path.basename(p))
    key = (0 if '/datasets/' in p else 1, p.count('/'), -os.path.getmtime(p))
    if name not in cands or key < cands[name][0]:
        cands[name] = (key, p)

for name in NEEDED:
    if name in cands:
        shutil.copy(cands[name][1], os.path.join(BIN, name))
        print(f"  {name:26s} <- {cands[name][1]}")

missing = [n for n in NEEDED if n not in cands]
assert not missing, ("scripts not found in the attached inputs: "
                     + ", ".join(missing)
                     + " -- check the dataset version contains the .py files")

path = f"{BIN}/oe62_dataset.py"
src = open(path).read()
if 'n_bad_smiles' not in src:
    old = """            smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            if smiles is not None:
                all_smiles.append(smiles)"""
    new = """            try:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            except Exception:
                n_bad_smiles += 1
                continue
            if smiles is not None:
                all_smiles.append(smiles)"""
    src = src.replace(old, new, 1)
    src = src.replace("        data_list = []\n        all_smiles = []",
                      "        data_list = []\n        all_smiles = []\n        n_bad_smiles = 0", 1)
    src = src.replace('''print(f"  {self.split}: {len(data_list)} graphs built")''',
                      '''print(f"  {self.split}: {len(data_list)} graphs built "
              f"({n_bad_smiles} skipped, bad SMILES)")''', 1)
    open(path, 'w').write(src)
    print("patched oe62_dataset.py")

os.makedirs(PKG, exist_ok=True)
open(f"{PKG}/__init__.py", "w").close()
shutil.copy(path, PKG)
!rm -rf {PKG}/__pycache__

# ---- the four upstream patches ---------------------------------------
def patch(fp, old, new, tag):
    s = open(fp).read()
    if tag in s:
        print("already patched:", os.path.basename(fp)); return
    assert old in s, f"anchor not found in {fp}"
    open(fp, 'w').write(s.replace(old, new, 1))
    print("patched:", os.path.basename(fp))

E = f"{REPO}/eqgat_diff/experiments"

patch(f"{E}/hparams.py",
      '            "crossdocked",',
      '            "crossdocked",\n            "oe62",', '"oe62"')

patch(f"{E}/run_train.py",
      '    elif hparams.dataset == "crossdocked":',
      '''    elif hparams.dataset == "oe62":
        dataset = "oe62"
        from experiments.data.oe62.oe62_dataset import OE62DataModule as DataModule
        non_adaptive = False

    elif hparams.dataset == "crossdocked":''', 'oe62')

patch(f"{E}/run_evaluation.py",
      '    elif hparams.dataset == "pubchem":',
      '''    elif hparams.dataset == "oe62":
        dataset = "oe62"
        from experiments.data.oe62.oe62_dataset import OE62DataModule as DataModule

    elif hparams.dataset == "pubchem":''', 'oe62')

di = f"{E}/data/data_info.py"
s = open(di).read()
if 'full_atom_encoder_oe62' not in s:
    old = '''        self.statistics = datamodule.statistics
        self.name = "drugs"
        self.atom_encoder = full_atom_encoder'''
    new = '''        self.statistics = datamodule.statistics
        if getattr(cfg, "dataset", None) == "oe62":
            self.name = "oe62"
            self.atom_encoder = full_atom_encoder_oe62
        else:
            self.name = "drugs"
            self.atom_encoder = full_atom_encoder'''
    assert old in s
    s = s.replace(old, new, 1) + '''

full_atom_encoder_oe62 = {
    "H": 0, "Li": 1, "B": 2, "C": 3, "N": 4, "O": 5, "F": 6, "Si": 7,
    "P": 8, "S": 9, "Cl": 10, "As": 11, "Se": 12, "Br": 13, "Te": 14, "I": 15,
}
'''
    open(di, 'w').write(s); print("patched: data_info.py")
else:
    print("already patched: data_info.py")

print()
!PYTHONPATH= {CONDA} run -n eqgatdiff python -c "import torch, rdkit, ase; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
!grep -c "oe62" {E}/hparams.py {E}/run_train.py {E}/run_evaluation.py {E}/data/data_info.py


# ====================================================================
# CELL 2 - restore full OE62, the generator and oracle 1
# ====================================================================

import time, shutil, glob, os
T0 = time.time()

# The previous commit's output holds the ONLY full copy of OE62. Identify it
# by size: an oe62_train.sdf over 100 MB is the real one, a few hundred KB is
# the biased mini-dataset a biasing run leaves behind.
cand = glob.glob('/kaggle/input/**/eqgat-diff/data/oe62/raw/oe62_train.sdf',
                 recursive=True)
cand = [p for p in cand if os.path.getsize(p) > 100 * 2**20]
assert cand, "full OE62 oe62_train.sdf (~145 MB) not found in the inputs"
GOOD = os.path.dirname(os.path.dirname(cand[0]))
print("full OE62 at:", GOOD)

!rm -rf {DROOT}/raw {DROOT}/processed
!cp -r {GOOD}/raw {DROOT}/
!cp -r {GOOD}/processed {DROOT}/
!ls -la {DROOT}/raw
!ls {DROOT}/processed | wc -l

ck   = glob.glob('/kaggle/input/**/runs/oe62_full70/run0/last.ckpt', recursive=True)[0]
orc1 = glob.glob('/kaggle/input/**/oracle/oracle_gap.pt', recursive=True)[0]
os.makedirs(f"{WORK}/runs/oe62_full70/run0", exist_ok=True)
os.makedirs(f"{WORK}/oracle", exist_ok=True)
shutil.copy(ck,   f"{WORK}/runs/oe62_full70/run0/last.ckpt")
shutil.copy(orc1, f"{WORK}/oracle/oracle_gap.pt")

CKPT    = f"{WORK}/runs/oe62_full70/run0/last.ckpt"
ORACLE  = f"{WORK}/oracle/oracle_gap.pt"
ORACLE2 = f"{WORK}/oracle2/oracle_gap.pt"     # trained below, in cell 5
print("\ngenerator:", os.path.getsize(CKPT)//2**20, "MB")
print("oracle 1 :", os.path.getsize(ORACLE)//2**10, "KB")


# ====================================================================
# CELL 3 - patch oe62_dataset.py (must run before any sampling)
# ====================================================================

def fix_datamodule(path):
    lines = open(path).read().splitlines(keepends=True)
    if any("self.train_dataset = train_dataset" in l for l in lines):
        print("already patched:", path)
        return
    anchor = next((i for i, l in enumerate(lines)
                   if '"test": test_dataset.statistics' in l), None)
    assert anchor is not None, f"anchor not found in {path}"
    j = anchor + 1
    while j < len(lines) and lines[j].strip() != "}":
        j += 1
    assert j < len(lines), f"closing brace not found in {path}"
    lines.insert(j + 1,
        "\n"
        "        self.train_dataset = train_dataset\n"
        "        self.val_dataset = val_dataset\n"
        "        self.test_dataset = test_dataset\n")
    open(path, "w").write("".join(lines))
    print("patched:", path)

for p in [f"{BIN}/oe62_dataset.py", f"{PKG}/oe62_dataset.py"]:
    fix_datamodule(p)

!rm -rf {PKG}/__pycache__
!grep -n "self.train_dataset" {PKG}/oe62_dataset.py


# ====================================================================
# CELL 4 - homo/lumo statistics: are IP and EA reachable?  ~3 min
# ====================================================================

%%time
code = r'''
import numpy as np
from ase.db import connect
db = connect("DBPATH")
h, l, g = [], [], []
for row in db.select():
    d = row.data
    h.append(float(np.ravel(d["homo"])[0]))
    l.append(float(np.ravel(d["lumo"])[0]))
    g.append(float(np.ravel(d["hlgap"])[0]))
h, l, g = map(np.array, (h, l, g))
for name, a in [("homo", h), ("lumo", l), ("hlgap", g)]:
    print(f"{name:6s} n={len(a)} mean={a.mean():10.3f} std={a.std():7.3f} "
          f"min={a.min():10.3f} max={a.max():10.3f}")
print(f"\nlumo-homo vs -hlgap: max abs diff = {np.abs((l-h)+g).max():.6f}")
'''
open("/kaggle/working/homo_stats.py","w").write(code.replace("DBPATH", DB))
!PYTHONPATH= {CONDA} run --no-capture-output -n eqgatdiff python /kaggle/working/homo_stats.py


# ====================================================================
# CELL 5 - retrain oracle 2 on the FULL set (--limit 60000)  ~2 h
# ====================================================================

%%time
!PYTHONPATH= {CONDA} run --no-capture-output -n eqgatdiff python {BIN}/train_oracle.py \
    --db {DB} \
    --label-key hlgap --negate --target gap \
    --max-atoms 70 --limit 60000 --epochs 120 --seed 2 \
    --out-dir {WORK}/oracle2

!ls -la {WORK}/oracle2


# ====================================================================
# CELL 6 - the biasing experiment: 5 loops x 3000  ~7.7 h
# ====================================================================

%%time
!PYTHONPATH= {CONDA} run --no-capture-output -n eqgatdiff python {BIN}/run_biasing_loop.py \
    --repo {REPO} --dataset-root {DROOT} \
    --start-ckpt {CKPT} \
    --oracle {ORACLE} --oracle2 {ORACLE2} \
    --work-dir {WORK}/biasing \
    --target gap --direction min \
    --n-loops 5 --n-sample 3000 --sample-batch 20 \
    --sigma-x 0.5 --finetune-epochs 10 --min-selected 20 \
    --select-from bonded

!cat {WORK}/biasing/history.json


# ====================================================================
# CELL 7 - Paper 1 section 2.3 analysis  ~minutes
# ====================================================================

loops = sorted(glob.glob(f"{WORK}/biasing/loop*/samples"))
print("sample sets found:", loops)

if loops:
    gen_args = " ".join(f"--generated {p}" for p in loops)
    names    = ",".join(os.path.basename(os.path.dirname(p)) for p in loops)
    !PYTHONPATH= {CONDA} run --no-capture-output -n eqgatdiff python {BIN}/analyze_generated.py \
        --reference {GOOD}/raw/oe62_train.sdf \
        {gen_args} --names {names} \
        --out-dir {WORK}/analysis
else:
    print("no sample directories -- the biasing loop produced none")


# ====================================================================
# CELL 8 - package the output
# ====================================================================

!rm -rf {DROOT}/raw {DROOT}/processed
!rm -f {WORK}/runs/oe62_full70/run0/epoch=*.ckpt

if not glob.glob('/kaggle/input/**/eqgatdiff_env.tar.gz', recursive=True) \
   and not os.path.exists(f"{WORK}/eqgatdiff_env.tar.gz"):
    !tar -czf {WORK}/eqgatdiff_env.tar.gz -C /opt/miniconda3/envs eqgatdiff
else:
    print("env tarball already in the inputs -- not re-tarring")

print(f"\nelapsed: {(time.time()-T0)/3600:.2f} h")
!du -sh {WORK}/* | sort -h
