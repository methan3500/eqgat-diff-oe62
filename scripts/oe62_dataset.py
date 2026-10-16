"""
oe62_dataset.py

STEP 2 of adding OE62 support to EQGAT-diff.

Place this file at:
    eqgat-diff/eqgat_diff/experiments/data/oe62/oe62_dataset.py
(and create an empty __init__.py alongside it)

This is modeled directly on EQGAT-diff's own qm9_dataset.py, with three
differences:
  1. It reads the SDF files produced by prepare_oe62.py instead of
     downloading QM9 from the internet (no download() method needed).
  2. full_atom_encoder covers OE62's 16 elements instead of QM9's set.
  3. No target/property dataframe — we only need structures, since the
     electronic properties come from SchNet+H later in the biasing loop,
     not from the dataset itself.
"""

import os.path as osp
from typing import Any, Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from experiments.data.abstract_dataset import (
    AbstractAdaptiveDataModule,
)
from experiments.data.metrics import compute_all_statistics
from experiments.data.utils import (
    Statistics,
    load_pickle,
    mol_to_torch_geometric,
    remove_hydrogens,
    save_pickle,
    train_subset,
)


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    return [value]


# OE62's 16 elements. Indices are contiguous from 0 as the model expects.
# Compare with EQGAT-diff's QM9 encoder, which shares 14 of these 16 and
# additionally has Al/Hg/Bi (absent from OE62) but lacks Li/Se/Te.
# Largest molecule the generator was trained on (prepare_oe62_sharded.py's
# --max-atoms). Every processed split pads its node-count histogram to this
# so `empirical_num_nodes` has a fixed size and fine-tuning on a small
# selection can still load the original checkpoint.
NUM_NODES_MAX = 70


full_atom_encoder = {
    "H": 0,
    "Li": 1,
    "B": 2,
    "C": 3,
    "N": 4,
    "O": 5,
    "F": 6,
    "Si": 7,
    "P": 8,
    "S": 9,
    "Cl": 10,
    "As": 11,
    "Se": 12,
    "Br": 13,
    "Te": 14,
    "I": 15,
}


class OE62Dataset(InMemoryDataset):
    def __init__(
        self,
        split,
        root,
        remove_h: bool = False,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        only_stats: bool = False,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.file_idx = {"train": 0, "val": 1, "test": 2}[split]
        self.remove_h = remove_h

        self.atom_encoder = full_atom_encoder
        if remove_h:
            self.atom_encoder = {
                k: v - 1 for k, v in self.atom_encoder.items() if k != "H"
            }

        super().__init__(root, transform, pre_transform, pre_filter)

        if not only_stats:
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            self.data, self.slices = None, None

        self.statistics = Statistics(
            num_nodes=load_pickle(self.processed_paths[1]),
            atom_types=torch.from_numpy(np.load(self.processed_paths[2])).float(),
            bond_types=torch.from_numpy(np.load(self.processed_paths[3])).float(),
            charge_types=torch.from_numpy(np.load(self.processed_paths[4])).float(),
            valencies=load_pickle(self.processed_paths[5]),
            bond_lengths=load_pickle(self.processed_paths[6]),
            bond_angles=torch.from_numpy(np.load(self.processed_paths[7])).float(),
            is_aromatic=torch.from_numpy(np.load(self.processed_paths[9])).float(),
            is_in_ring=torch.from_numpy(np.load(self.processed_paths[10])).float(),
            hybridization=torch.from_numpy(np.load(self.processed_paths[11])).float(),
        )
        self.smiles = load_pickle(self.processed_paths[8])

    @property
    def raw_file_names(self):
        # These are produced by prepare_oe62.py and must sit in <root>/raw/
        return ["oe62_train.sdf", "oe62_val.sdf", "oe62_test.sdf"]

    @property
    def processed_file_names(self):
        h = "noh" if self.remove_h else "h"
        s = self.split
        return [
            f"{s}_{h}.pt",
            f"{s}_n_{h}.pickle",
            f"{s}_atom_types_{h}.npy",
            f"{s}_bond_types_{h}.npy",
            f"{s}_charges_{h}.npy",
            f"{s}_valency_{h}.pickle",
            f"{s}_bond_lengths_{h}.pickle",
            f"{s}_angles_{h}.npy",
            f"{s}_smiles.pickle",
            f"{s}_is_aromatic_{h}.npy",
            f"{s}_is_in_ring_{h}.npy",
            f"{s}_hybridization_{h}.npy",
        ]

    def download(self):
        # Nothing to download: prepare_oe62.py generates the raw SDF files.
        missing = [p for p in self.raw_paths if not osp.exists(p)]
        if missing:
            raise FileNotFoundError(
                "Missing OE62 raw files:\n  "
                + "\n  ".join(missing)
                + "\n\nRun prepare_oe62.py first to generate them."
            )

    def process(self):
        RDLogger.DisableLog("rdApp.*")

        raw_path = self.raw_paths[self.file_idx]
        suppl = Chem.SDMolSupplier(raw_path, removeHs=False, sanitize=True)

        data_list = []
        all_smiles = []
        n_unreadable = 0
        n_bad_smiles = 0
        n_bad_graph = 0

        for mol in tqdm(suppl, desc=f"Processing OE62 {self.split}"):
            if mol is None:
                n_unreadable += 1
                continue

            # A small number of molecules sanitise successfully but fail
            # canonical SMILES generation with "bad bond type" — bond
            # perception produced a bond RDKit will not canonicalise, usually
            # an aromatic flag that did not survive the molblock round trip.
            # Skip those individually rather than losing the whole build.
            try:
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
            except Exception:
                n_bad_smiles += 1
                continue

            if smiles is not None:
                all_smiles.append(smiles)

            try:
                data = mol_to_torch_geometric(mol, full_atom_encoder, smiles)
            except Exception:
                n_bad_graph += 1
                continue

            if self.remove_h:
                data = remove_hydrogens(data)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        print(f"  {self.split}: {len(data_list)} graphs built "
              f"(skipped: {n_unreadable} unreadable, {n_bad_smiles} bad SMILES, "
              f"{n_bad_graph} graph conversion failures)")
        torch.save(self.collate(data_list), self.processed_paths[0])

        statistics = compute_all_statistics(
            data_list,
            self.atom_encoder,
            charges_dic={-2: 0, -1: 1, 0: 2, 1: 3, 2: 4, 3: 5},
            additional_feats=True,
        )

        # Pad the node-count histogram so every build spans the same support.
        #
        # The model registers `empirical_num_nodes` with a size taken from the
        # TRAIN statistics' node counts. A biasing fine-tune runs on a small
        # selection whose largest molecule may have far fewer atoms than the
        # original training set, which produces a smaller buffer and makes
        # checkpoint loading fail with:
        #     size mismatch for empirical_num_nodes:
        #     copying a param with shape torch.Size([71]) ... current [53]
        # Covering 0..NUM_NODES_MAX with zero counts keeps the buffer the same
        # size for every subset, without changing any observed frequency.
        num_nodes = statistics.num_nodes
        try:
            for n in range(0, NUM_NODES_MAX + 1):
                if n not in num_nodes:
                    num_nodes[n] = 0
            print(f"  {self.split}: node-count histogram padded to "
                  f"0..{NUM_NODES_MAX} ({len(num_nodes)} bins)")
        except TypeError:
            print(f"  {self.split}: num_nodes is not a mapping "
                  f"({type(num_nodes).__name__}); left unpadded")
        save_pickle(num_nodes, self.processed_paths[1])
        np.save(self.processed_paths[2], statistics.atom_types)
        np.save(self.processed_paths[3], statistics.bond_types)
        np.save(self.processed_paths[4], statistics.charge_types)
        save_pickle(statistics.valencies, self.processed_paths[5])
        save_pickle(statistics.bond_lengths, self.processed_paths[6])
        np.save(self.processed_paths[7], statistics.bond_angles)
        save_pickle(set(all_smiles), self.processed_paths[8])
        np.save(self.processed_paths[9], statistics.is_aromatic)
        np.save(self.processed_paths[10], statistics.is_in_ring)
        np.save(self.processed_paths[11], statistics.hybridization)


class OE62DataModule(AbstractAdaptiveDataModule):
    def __init__(self, cfg, only_stats: bool = False):
        self.datadir = cfg.dataset_root
        self.cfg = cfg
        root_path = cfg.dataset_root

        train_dataset = OE62Dataset(
            split="train", root=root_path,
            remove_h=cfg.remove_hs, only_stats=only_stats,
        )
        val_dataset = OE62Dataset(
            split="val", root=root_path,
            remove_h=cfg.remove_hs, only_stats=only_stats,
        )
        test_dataset = OE62Dataset(
            split="test", root=root_path,
            remove_h=cfg.remove_hs, only_stats=only_stats,
        )

        self.statistics = {
            "train": train_dataset.statistics,
            "val": val_dataset.statistics,
            "test": test_dataset.statistics,
        }

        # run_evaluation.py constructs this datamodule with only_stats=True and
        # then reads datamodule.train_dataset.smiles. In that path we never
        # reach super().__init__() below, which is what would normally set
        # these attributes, so assign them here explicitly.
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        if not only_stats:
            self.remove_h = cfg.remove_hs
            self.train_smiles = train_dataset.smiles
            self.val_smiles = val_dataset.smiles
            self.test_smiles = test_dataset.smiles

            if cfg.select_train_subset:
                self.idx = train_subset(
                    dset_len=len(train_dataset),
                    train_size=cfg.train_size,
                    seed=cfg.seed,
                    filename=f"{self.datadir}/splits.npz",
                )
                self.train_smiles = train_dataset.smiles
                train_dataset = torch.utils.data.Subset(train_dataset, self.idx)

            super().__init__(cfg, train_dataset, val_dataset, test_dataset)

    def _train_dataloader(self, shuffle=True):
        return super()._train_dataloader(shuffle=shuffle)

    def compute_mean_mad(self, properties_list):
        # OE62 structures carry no regression targets here; electronic
        # properties come from SchNet+H downstream in the biasing loop.
        raise NotImplementedError(
            "OE62DataModule has no built-in regression targets."
        )
