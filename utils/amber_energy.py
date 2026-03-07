import argparse
import os, sys
# os.chdir('/data/xiwei/decomp_data_augmentation')
# sys.path.append('/data/xiwei/decomp_data_augmentation')

from tqdm import tqdm
from collections import defaultdict
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
import numpy as np
import torch

import utils.misc as misc
from utils.transforms import MAP_ATOM_TYPE_ONLY_TO_INDEX
from datasets.pl_pair_dataset import get_decomp_dataset

def stat_amber_info(config='configs/preprocessing/amber_energy.yml',
                    bond_info_path='data/bond_stat.pt', angle_info_path='data/angle_stat.pt'):
    config = misc.load_config(config)
    dataset, subsets = get_decomp_dataset(
        config=config.data,
        transform=None,
    )
    train_set, val_set = subsets['train'], subsets['test']

    bond_info = defaultdict(list)
    angle_info = defaultdict(list)
    for data in tqdm(train_set):
        rdmol = data.ligand_rdmol
        conf = rdmol.GetConformer()

        for bond in rdmol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            atom1_type = atom1.GetSymbol()
            atom2_type = atom2.GetSymbol()
            atom1_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(atom1_type)]
            atom2_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(atom2_type)]

            bond_length = rdMolTransforms.GetBondLength(conf, atom1.GetIdx(), atom2.GetIdx())
            if atom1_idx < atom2_idx:
                bond_info[(atom1_idx, atom2_idx)].append(bond_length)
            else:
                bond_info[(atom2_idx, atom1_idx)].append(bond_length)

        for atom in rdmol.GetAtoms():
            nbrs = [nbr for nbr in atom.GetNeighbors()]
            if len(nbrs) > 1:
                atom_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(atom.GetSymbol())]
                for nb1, nb2 in combinations(nbrs, 2):
                    nb1_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(nb1.GetSymbol())]
                    nb2_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(nb2.GetSymbol())]
                    ang = rdMolTransforms.GetAngleDeg(conf, nb1.GetIdx(), atom.GetIdx(), nb2.GetIdx())
                    if nb1_idx < nb2_idx:
                        angle_info[(nb1_idx, atom_idx, nb2_idx)].append(ang)
                    else:
                        angle_info[(nb2_idx, atom_idx, nb1_idx)].append(ang)

    torch.save(bond_info, bond_info_path)
    torch.save(angle_info, angle_info_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/preprocessing/amber_energy.yml')
    args = parser.parse_args()

    stat_amber_info(args.config)