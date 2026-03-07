import os, sys
# os.chdir('/data/xiwei/decomp_data_augmentation')
# sys.path.append('/data/xiwei/decomp_data_augmentation')

from copy import deepcopy
import itertools
import os
from statistics import mode
import numpy as np
import pickle
from collections import defaultdict
import random
from itertools import product

import lmdb
import torch
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from torch.utils.data import Dataset, Subset
from tqdm.auto import tqdm

from utils.data import PDBProtein
from utils.data import ProteinLigandData, torchify_dict, parse_sdf_file
from utils.prior import compute_golden_prior_from_data
import utils.transforms as trans

INF = 9999999

def get_submol_from_mol(src_mol, atom_indices):

    assert isinstance(src_mol, Chem.Mol)
    positions = src_mol.GetConformer().GetPositions()

    emol = Chem.RWMol()
    id_map = {}
    for i,a_id in enumerate(atom_indices):
        emol.AddAtom(src_mol.GetAtomWithIdx(int(a_id)))
        id_map[a_id] = i
        
    for bond in src_mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if start in atom_indices and end in atom_indices:
            emol.AddBond(id_map[start],id_map[end],bond.GetBondType())
    
    rdmol = emol.GetMol()

    assert rdmol.GetNumAtoms() == positions[atom_indices].shape[0]
    submol_pos = positions[atom_indices]
    conf = Chem.Conformer(rdmol.GetNumAtoms())
    for i in range(submol_pos.shape[0]):
        conf.SetAtomPosition(i, submol_pos[i].tolist())
    rdmol.AddConformer(conf, assignId=True)

    return rdmol 


def recon_mol_from_sample(res):
    pred_pos = res['pred_pos']
    pred_v = res['pred_v']

    pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='basic')
    mw = Chem.RWMol()
    for a in pred_atom_type:
        mw.AddAtom(Chem.Atom(a))
        
    bond_type = res['pred_bond_type']
    bond_index = res['pred_bond_index']
    for idx, bond_order in enumerate(bond_type):
        b1 = int(bond_index[0][idx])
        b2 = int(bond_index[1][idx])
        if b1 < b2:
            if bond_order == 0:
                continue
            if bond_order == 1:
                mw.AddBond(b1, b2, Chem.BondType.SINGLE)
            elif bond_order == 2:
                mw.AddBond(b1, b2, Chem.BondType.DOUBLE)
            elif bond_order == 3:
                mw.AddBond(b1, b2, Chem.BondType.TRIPLE)
            elif bond_order == 4:
                mw.AddBond(b1, b2, Chem.BondType.AROMATIC)
            else:
                raise ValueError

    mol = mw.GetMol()
    try:
        Chem.SanitizeMol(mol)
    except:
        pass
    conf = Chem.Conformer()
    for i in range(mol.GetNumAtoms()):
        x,y,z = pred_pos[i]
        conf.SetAtomPosition(i,Point3D(x,y,z))
    mol.AddConformer(conf)
    try:
        smiles = Chem.MolToSmiles(mol)
    except:
        smiles = ''

    pred_pos = pred_pos.astype(np.float32)
    ptable = Chem.GetPeriodicTable()
    accum_pos = 0
    accum_mass = 0
    for atom_idx in range(mol.GetNumAtoms()):
        atom = mol.GetAtomWithIdx(atom_idx)
        atom_num = atom.GetAtomicNum()
        atom_weight = ptable.GetAtomicWeight(atom_num)
        accum_pos += pred_pos[atom_idx] * atom_weight
        accum_mass += atom_weight
    center_of_mass = accum_pos / accum_mass

    return {
        'rdmol': mol,
        'element': np.array(pred_atom_type),
        'pos': pred_pos,
        'bond_index': np.array(bond_index)[:,np.nonzero(bond_type)[0]],
        'bond_type': bond_type[np.nonzero(bond_type)],
        'center_of_mass': center_of_mass,
        'smiles': smiles
    }


def get_decomp_dataset(config, **kwargs):
    name = config.name
    root = config.path
    if name == 'pl':
        dataset = DecompPLPairDataset(root, mode=config.mode,
                                      include_dummy_atoms=config.include_dummy_atoms, version=config.version,
                                      ori_pair_path=getattr(config, 'ori_pair_path', None), processed_pair_path=getattr(config, 'processed_pair_path', None),
                                      **kwargs)
    else:
        raise NotImplementedError('Unknown dataset: %s' % name)

    try:
        split = dataset.split_train_val_dataset()
        subsets = {}
        for k, v in split.items():
            high_set = DecompPLPairSubset(dataset, indices=v, load_high_data=True)
            low_set = DecompPLPairSubset(dataset, indices=v, load_high_data=False)
            
            subsets[k] = {'high_set': high_set, 'low_set': low_set}
        return dataset, subsets
    except Exception as e:
        print(e)
        return dataset


class DecompPLPairDataset(Dataset):

    def __init__(self, raw_path, transform=None, mode='full',
                 include_dummy_atoms=False, kekulize=True, version='v1', 
                 ori_pair_path=None, processed_pair_path='data/processed_pair.pt', protein_root='data/crossdocked_v1.1_rmsd1.0',
                 processed_protein_root='data/crossdocked_v1.1_rmsd1.0_processed', protein_radius=10):
        super().__init__()
        self.raw_path = raw_path.rstrip('/')
        self.mode = mode  # ['arms', 'scaffold', 'full']
        self.include_dummy_atoms = include_dummy_atoms
        self.kekulize = kekulize
        self.processed_path = os.path.join(os.path.dirname(self.raw_path),
                                           os.path.basename(self.raw_path) + f'_{mode}_{version}.lmdb')
        self.name2id_path = os.path.join(os.path.dirname(self.raw_path),
                                         os.path.basename(self.raw_path) + f'_{mode}_{version}_name2id.pt')

        self.transform = transform
        self.mode = mode
        self.db = None
        self.keys = None
        
        self.protein_root = protein_root
        self.processed_protein_root = processed_protein_root
        self.ori_pair_path = ori_pair_path
        self.processed_pair_path = processed_pair_path
        self.protein_radius = protein_radius
        
        self.__init_dataset__()
        

    def __init_dataset__(self):
        if not os.path.exists(self.processed_path):
            print(f'{self.processed_path} does not exist, begin processing data')
            self._process()
        print('Load dataset from %s' % self.processed_path)
        
        if not os.path.exists(self.name2id_path):
            self._precompute_name2id()
        self.name2id = torch.load(self.name2id_path)
        print('Load name2id from %s' % self.name2id_path)
        self.all_pairs = torch.load(self.processed_pair_path)
        print('Load pairs from %s' % self.processed_pair_path)

    def _connect_db(self):
        """
            Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))
            

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None
        

    def _precompute_name2id(self):
        name2id = defaultdict(list)
        for i in tqdm(range(self.__len__()), 'Indexing'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue

            try:
                name = data.src_ligand_filename[:-4]
            except:
                continue
            # name = (data.src_protein_filename, data.src_ligand_filename)
            name2id[name].append(i)
        torch.save(name2id, self.name2id_path)
        
    

    def split_train_val_dataset(self, val_num=0):
        if self.db is None:
            self._connect_db()

        train_val_split = np.ones(len(self.all_pairs))
        val_index = random.sample(range(len(train_val_split)), val_num)
        train_val_split[val_index] = 0
        return {
            'train': np.where(train_val_split == 1)[0],
            'test': np.where(train_val_split == 0)[0]
        }

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )
        pair_dict = torch.load(self.ori_pair_path)

        num_skipped = 0
        num_data = 0
        recon_fail = 0
        all_pairs = []
        with db.begin(write=True, buffers=True) as txn:
            for idx, pairs in tqdm(pair_dict.items()):
                for p in pairs:
                    pair_key = {}
                    for k, r in p.items():
                        try:
                            ori_ligand_file = r['ligand_filename']
                            protein_path = os.path.join(self.protein_root, \
                                                    ori_ligand_file.split('/')[0], ori_ligand_file.split('/')[1][:10] + '.pdb')
                            pocket_path = os.path.join(self.processed_protein_root, \
                                                    ori_ligand_file.split('/')[0], ori_ligand_file.split('/')[1][:-4] + '_pocket.pdb')
                            num_arms, num_scaffold = max(r['decomp_mask']) + 1, int(-1 in r['decomp_mask'])
                            ligand_file = ori_ligand_file
                            mol = r['mol']
                            
                            if self.mode == 'full':
                                protein = PDBProtein(pocket_path)
                                protein_dict = protein.to_dict_atom()
                                if mol is None:
                                    ligand_dict = recon_mol_from_sample(r)
                                    mol = ligand_dict['rdmol']
                                else:
                                    ligand_dict = parse_sdf_file(mol, kekulize=self.kekulize)
                                num_protein_atoms, num_ligand_atoms = len(protein.atoms), ligand_dict['rdmol'].GetNumAtoms()

                                # extract ligand arms & atom mask
                                ligand_atom_mask = torch.tensor(r['decomp_mask'])
                                arms = []
                                for arm_idx in range(num_arms):
                                    atom_indices = torch.where(ligand_atom_mask == arm_idx)[0].tolist()

                                    arm = get_submol_from_mol(mol, atom_indices)
                                    if arm is None:
                                        print(f"[fail] to extract submol (arm).")
                                        raise ValueError
                                    smi = Chem.MolToSmiles(arm)
                                    if r['mol'] is not None:
                                        if "." in smi:
                                            print(f"[fail] incompleted arm: {smi}")
                                            raise ValueError
                                    arms.append(arm)
                                
                                # extract pocket atom mask
                                protein_atom_serial = [atom['atom_id'] for atom in protein.atoms]
                                pocket_atom_masks = []
                                for arm in arms:
                                    selected_atom_serial, union_residues = protein.query_residues_centers(arm.GetConformer(0).GetPositions(), self.protein_radius)
                                    pocket_atom_idx = [protein_atom_serial.index(i) for i in selected_atom_serial]
                                    pocket_atom_mask = torch.zeros(num_protein_atoms, dtype=torch.bool)
                                    pocket_atom_mask[pocket_atom_idx] = 1
                                    pocket_atom_masks.append(pocket_atom_mask)
                                pocket_atom_masks = torch.stack(pocket_atom_masks)

                                data = ProteinLigandData.from_protein_ligand_dicts(
                                    protein_dict=torchify_dict(protein_dict),
                                    ligand_dict=torchify_dict(ligand_dict),
                                )
                                data.src_protein_filename = protein_path
                                data.src_ligand_filename = ori_ligand_file
                                data.num_arms, data.num_scaffold = num_arms, num_scaffold
                                data.pocket_atom_masks, data.ligand_atom_mask = pocket_atom_masks, ligand_atom_mask
                                data = compute_golden_prior_from_data(data)
                                if r['mol'] is None:
                                    data.protein_file, data.ligand_file = pocket_path, None
                                    # save property
                                    data.qed, data.sa = 0, 0
                                    data.score_only, data.vina_min = INF, INF
                                    data.arms_list = [{'arm': None, 'qed': 0, 'sa': 0, 'score_only': INF, 'vina_min': INF} for i in range(data.num_arms)]
                                    if data.num_scaffold > 0:
                                        data.scaffold_list = [{'arm': None, 'qed': 0, 'sa': 0, 'score_only': INF, 'vina_min': INF}]
                                    else:
                                        data.scaffold_list = []
                                    data.recon_succ = 0
                                else:
                                    data.protein_file, data.ligand_file = pocket_path, ligand_file
                                    # save property
                                    data.qed, data.sa = r['chem_results']['qed'], r['chem_results']['sa']
                                    data.score_only, data.vina_min = r['vina']['score_only'][0]['affinity'], r['vina']['minimize'][0]['affinity']
                                    if data.num_scaffold > 0:
                                        data.scaffold_list = [r['arms_dict'][-1]]
                                    else:
                                        data.scaffold_list = []
                                    assert len(r['arms_dict']) == data.num_arms + data.num_scaffold
                                    arms_list = [{} for i in range(data.num_arms)]
                                    for i in range(data.num_arms):
                                        arms_list[i] = r['arms_dict'][i]
                                    data.arms_list = arms_list
                                    data.recon_succ = 1
                                
                                data = data.to_dict()  # avoid torch_geometric version issue
                                txn.put(
                                    # key=str(num_data).encode(),
                                    key=f'{num_data:08d}'.encode(),
                                    value=pickle.dumps(data)
                                )
                            else:
                                raise ValueError
                            pair_key[k] = f'{num_data:08d}'.encode()
                            num_data += 1
                            if r['mol'] is None:
                                recon_fail += 1
                        except Exception as e:
                            print(e)
                            num_skipped += 1
                            print('Skipping (%d) %s gen %d' % (num_skipped, k, idx))
                            continue
                    if len(pair_key) == 2:
                        all_pairs.append(pair_key)
        db.close()
        self.all_pairs = all_pairs
        print(f'total pairs num: {len(all_pairs)}, recon fail pairs num: {recon_fail}')
        torch.save(self.all_pairs, self.processed_pair_path)
        

    def __len__(self):
        if self.db is None:
            self._connect_db()
            
        return len(self.keys)
    

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()

        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
        
        # assert data.protein_pos.size(0) > 0
        if self.transform is not None:
            data = self.transform(data)
        # exclude keys added to uniform data
        if getattr(self, 'exclude_keys', None) is not None:
            new_data = deepcopy(data)
            for k in self.exclude_keys:
                if k in new_data:
                    delattr(new_data, k)
            return new_data
        return data
    
class DecompPLPairSubset(DecompPLPairDataset):
    def __init__(self, dataset, indices, load_high_data=True):
        self.dataset = dataset
        self.indices = indices
        if self.dataset.db is None:
            self.dataset._connect_db()
        self.keys = [self.dataset.keys[i] for i in indices]
        self.load_high_data = load_high_data
        self.all_pairs = dataset.all_pairs
        
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        if self.dataset.db is None:
            self.dataset._connect_db()
        
        key_pair = self.all_pairs[idx]
        if self.load_high_data:
            key = key_pair['high']
        else:
            key = key_pair['low']
        data = pickle.loads(self.dataset.db.begin().get(key))
        data = ProteinLigandData(**data)
        data.id = idx
            
        # assert data.protein_pos.size(0) > 0
        if self.dataset.transform is not None:
            data = self.dataset.transform(data)
        # exclude keys added to uniform data
        if getattr(self, 'exclude_keys', None) is not None:
            new_data = deepcopy(data)
            for k in self.dataset.exclude_keys:
                if k in new_data:
                    delattr(new_data, k)
            return new_data
        return data
        

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--mode', type=str, default='full')
    parser.add_argument('--dummy', type=eval, default=False)
    parser.add_argument('--keku', type=eval, default=True)
    parser.add_argument('--version', type=str, default='ref_prior_aromatic_decompdpo')
    parser.add_argument('--ori_pair_path', type=str, default='data/training_pairs.pt')
    parser.add_argument('--processed_pair_path', type=str, default='data/processed_pairs.pt')
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')
    dataset = DecompPLPairDataset(args.path, mode=args.mode,
                                  include_dummy_atoms=args.dummy, kekulize=args.keku, 
                                  version=args.version, ori_pair_path=args.ori_pair_path, processed_pair_path=args.processed_pair_path)
    dataset.__init_dataset__()
    print(len(dataset), dataset[0])
