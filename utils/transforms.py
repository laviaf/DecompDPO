import numpy as np
import torch
import torch.nn.functional as F
import math
from itertools import combinations

from rdkit import Chem
from rdkit.Chem import rdMolTransforms

from utils import data as utils_data
from utils.data import ProteinLigandData

AROMATIC_FEAT_MAP_IDX = utils_data.ATOM_FAMILIES_ID['Aromatic']

# only atomic number 1, 6, 7, 8, 9, 15, 16, 17 exist
MAP_ATOM_TYPE_FULL_TO_INDEX = {
    (1, 'S', False): 0,
    (6, 'SP', False): 1,
    (6, 'SP2', False): 2,
    (6, 'SP2', True): 3,
    (6, 'SP3', False): 4,
    (7, 'SP', False): 5,
    (7, 'SP2', False): 6,
    (7, 'SP2', True): 7,
    (7, 'SP3', False): 8,
    (8, 'SP2', False): 9,
    (8, 'SP2', True): 10,
    (8, 'SP3', False): 11,
    (9, 'SP3', False): 12,
    (15, 'SP2', False): 13,
    (15, 'SP2', True): 14,
    (15, 'SP3', False): 15,
    (15, 'SP3D', False): 16,
    (16, 'SP2', False): 17,
    (16, 'SP2', True): 18,
    (16, 'SP3', False): 19,
    (16, 'SP3D', False): 20,
    (16, 'SP3D2', False): 21,
    (17, 'SP3', False): 22
}

MAP_ATOM_TYPE_ONLY_TO_INDEX = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4,
    15: 5,
    16: 6,
    17: 7,
}

MAP_ATOM_TYPE_AROMATIC_TO_INDEX = {
    (1, False): 0,
    (6, False): 1,
    (6, True): 2,
    (7, False): 3,
    (7, True): 4,
    (8, False): 5,
    (8, True): 6,
    (9, False): 7,
    (15, False): 8,
    (15, True): 9,
    (16, False): 10,
    (16, True): 11,
    (17, False): 12
}

MAP_INDEX_TO_ATOM_TYPE_ONLY = {v: k for k, v in MAP_ATOM_TYPE_ONLY_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_AROMATIC = {v: k for k, v in MAP_ATOM_TYPE_AROMATIC_TO_INDEX.items()}
MAP_INDEX_TO_ATOM_TYPE_FULL = {v: k for k, v in MAP_ATOM_TYPE_FULL_TO_INDEX.items()}


def get_atomic_number_from_index(index, mode):
    if mode == 'basic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_ONLY[i] for i in index.tolist()]
    elif mode == 'add_aromatic':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][0] for i in index.tolist()]
    elif mode == 'full':
        atomic_number = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][0] for i in index.tolist()]
    else:
        raise ValueError
    return atomic_number


def is_aromatic_from_index(index, mode):
    if mode == 'add_aromatic':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    elif mode == 'full':
        is_aromatic = [MAP_INDEX_TO_ATOM_TYPE_FULL[i][2] for i in index.tolist()]
    elif mode == 'basic':
        is_aromatic = None
    else:
        raise ValueError
    return is_aromatic


def get_hybridization_from_index(index, mode):
    if mode == 'full':
        hybridization = [MAP_INDEX_TO_ATOM_TYPE_AROMATIC[i][1] for i in index.tolist()]
    else:
        raise ValueError
    return hybridization


def get_index(atom_num, hybridization, is_aromatic, mode):
    if mode == 'basic':
        return MAP_ATOM_TYPE_ONLY_TO_INDEX[int(atom_num)]
    elif mode == 'add_aromatic':
        return MAP_ATOM_TYPE_AROMATIC_TO_INDEX[int(atom_num), bool(is_aromatic)]
    else:
        return MAP_ATOM_TYPE_FULL_TO_INDEX[(int(atom_num), str(hybridization), bool(is_aromatic))]


class FeaturizeProteinAtom(object):

    def __init__(self):
        super().__init__()
        self.atomic_numbers = torch.LongTensor([1, 6, 7, 8, 16, 34])  # H, C, N, O, S, Se
        self.max_num_aa = 20

    @property
    def protein_feature_dim(self):
        return self.atomic_numbers.size(0) + self.max_num_aa + 1

    def __call__(self, data: ProteinLigandData):
        element = data.protein_element.view(-1, 1) == self.atomic_numbers.view(1, -1)  # (N_atoms, N_elements)
        amino_acid = F.one_hot(data.protein_atom_to_aa_type, num_classes=self.max_num_aa)
        is_backbone = data.protein_is_backbone.view(-1, 1).long()
        x = torch.cat([element, amino_acid, is_backbone], dim=-1)
        data.protein_atom_feature = x
        return data


class FeaturizeLigandAtom(object):

    def __init__(self, mode='basic', prior_types=True, has_recon_failed_data=False):
        super().__init__()
        assert mode in ['basic', 'add_aromatic', 'full']
        self.mode = mode
        self.prior_types = prior_types
        if self.mode == 'basic' and self.prior_types:
            self.atom_types_prob = np.array([0., 0.6716, 0.1174, 0.1689, 0.01315, 0.01117, 0.01128, 0.00647])
            self.bond_types_prob = np.array([0.9170, 0.0433, 0.00687, 0.000173, 0.03266])
        else:
            self.atom_types_prob, self.bond_types_prob = None, None
        self.has_recon_failed_data = has_recon_failed_data

    @property
    def ligand_feature_dim(self):
        if self.mode == 'basic':
            return len(MAP_ATOM_TYPE_ONLY_TO_INDEX)
        elif self.mode == 'add_aromatic':
            return len(MAP_ATOM_TYPE_AROMATIC_TO_INDEX)
        else:
            return len(MAP_ATOM_TYPE_FULL_TO_INDEX)

    def __call__(self, data: ProteinLigandData):
        element_list = data.ligand_element
        if self.has_recon_failed_data:
            x = [MAP_ATOM_TYPE_ONLY_TO_INDEX[int(e)] for e in element_list]
        else:
            hybridization_list = data.ligand_hybridization
            aromatic_list = [v[AROMATIC_FEAT_MAP_IDX] for v in data.ligand_atom_feature]
            x = [get_index(e, h, a, self.mode) for e, h, a in zip(element_list, hybridization_list, aromatic_list)]
        x = torch.tensor(x)
        data.ligand_atom_feature_full = x
        return data


class ComputeLigandAtomNoiseDist(object):
    # for decomposing the drug space
    # Compute Ligand Atom Noise Distribution!
    def __init__(self, version):
        super().__init__()
        self.version = version
        print(f'{version} prior mode is applied in ComputeLigandAtomNoiseDist transform!')
        assert version in ['subpocket', 'ref_prior', 'beta_prior']

    def __call__(self, data: ProteinLigandData):
        """
        Func:
            data.pocket_atom_masks, data.protein_pos --> data.ligand_decomp_centers
            data.ligand_atom_mask --> data.ligand_decomp_mask
        """
        if self.version == 'subpocket':
            arm_centers = []
            for arm_idx, pocket_mask in enumerate(data.pocket_atom_masks):
                if pocket_mask.sum() > 0:
                    arm_centers.append(data.protein_pos[pocket_mask].mean(0))
                else:
                    # special case (data id: 86623)
                    arm_centers.append(data.ligand_pos[data.ligand_atom_mask == arm_idx].mean(0))
            data.arm_centers = torch.stack(arm_centers)
            data.scaffold_center = data.protein_pos.mean(0).unsqueeze(0)  # [1, 3]
            data.ligand_decomp_centers = torch.cat([data.arm_centers, data.scaffold_center], dim=0)
            data.ligand_decomp_stds = torch.ones_like(data.ligand_decomp_centers)

        elif self.version == 'ref_prior' or self.version == 'beta_prior':
            ## TODO: add some hyperparameter
            min_std = 0.6

            arms_centers = []
            scaffold_centers = []
            arms_std = []
            scaffold_std = []
            scaffold_prior = data.scaffold_prior
            arms_prior = data.arms_prior
            # >>> add arms prior
            for arm_id in range(data.num_arms):
                (arm_atom_num, arm_iso_mu, arm_iso_cov, arm_aniso_mu, arm_aniso_cov) = arms_prior[arm_id]
                if arm_atom_num > 1:
                    tmp_std = torch.tensor([torch.sqrt(arm_iso_cov[0, 0])]).reshape(1, 1).expand(-1, 3).clone()
                    tmp_std = torch.clamp(tmp_std, min=min_std)
                    arms_std.append(tmp_std)
                else:
                    tmp_std = torch.tensor([min_std]).reshape(1, 1).expand(-1, 3).clone()
                    arms_std.append(tmp_std)
                arms_centers.append(arm_iso_mu.unsqueeze(0))
            # >>> add scaffold prior
            if len(scaffold_prior) > 0:
                assert len(scaffold_prior) == 1
                assert len(scaffold_prior) == data.num_scaffold
                (scaffold_atom_num, scaffold_iso_mu, scaffold_iso_cov, scaffold_aniso_mu, scaffold_aniso_cov) = \
                scaffold_prior[0]
                scaffold_centers.append(scaffold_iso_mu.unsqueeze(0))
                if self.version == 'ref_prior':
                    if scaffold_atom_num > 1:
                        tmp_std = torch.tensor([torch.sqrt(scaffold_iso_cov[0, 0])]).reshape(1, 1).expand(-1, 3).clone()
                        tmp_std = torch.clamp(tmp_std, min=min_std)
                        scaffold_std.append(tmp_std)
                    elif scaffold_atom_num == 1:
                        tmp_std = torch.tensor([min_std]).reshape(1, 1).expand(-1, 3).clone()
                        scaffold_std.append(tmp_std)
                    else:
                        raise ValueError(f"scaffold_atom_num = {scaffold_atom_num}, not valid!")
                elif self.version == 'beta_prior':
                    # NOTE: scaffold_iso_cov of 'beta_prior' is scalar now, instead of np.eye
                    # NOTE: beta_prior does not set min clamp here
                    try:
                        tmp_std = torch.tensor([torch.sqrt(scaffold_iso_cov)]).reshape(1, 1).expand(-1, 3).clone()
                    except:
                        tmp_std = torch.tensor([torch.sqrt(scaffold_iso_cov[0, 0])]).reshape(1, 1).expand(-1, 3).clone()
                    tmp_std = torch.clamp(tmp_std, min=min_std)
                    scaffold_std.append(tmp_std)
            else:
                scaffold_centers.append(data.protein_pos.mean(0).unsqueeze(0))  # [1, 3]
                scaffold_std.append(torch.tensor([min_std]).reshape(1, 1).expand(-1, 3).clone())

            data.ligand_decomp_centers = torch.cat(arms_centers + scaffold_centers, dim=0)
            data.ligand_decomp_stds = torch.cat(arms_std + scaffold_std, dim=0).float()

        else:
            raise NotImplementedError

        data.arm_num_atoms = torch.tensor([(data.ligand_atom_mask == i).sum() for i in range(data.num_arms)])
        data.scaffold_num_atoms = (data.ligand_atom_mask == -1).sum().unsqueeze(0)
        data.ligand_decomp_num_atoms = torch.cat([data.arm_num_atoms, data.scaffold_num_atoms])
        return data


class AddDecompIndicator(object):
    def __init__(self, max_num_arms=10, global_prior_index=None, add_ord_feat=False,
                 add_to_protein=True, add_to_ligand=True):
        super().__init__()
        self.max_num_arms = max_num_arms
        self.global_prior_index = global_prior_index
        self.add_ord_feat = add_ord_feat
        self.num_classes = max_num_arms + 1  # + scaffold
        self.add_to_protein = add_to_protein
        self.add_to_ligand = add_to_ligand

    @property
    def protein_feature_dim(self):
        ndim = 2  # arm / scaffold indicator (2-dim)
        if self.add_ord_feat:
            ndim += self.max_num_arms + 1  # one-hot index
        return ndim

    @property
    def ligand_feature_dim(self):
        ndim = 2  # arm / scaffold indicator (2-dim)
        if self.add_ord_feat:
            ndim += self.max_num_arms + 1  # one-hot index
        return ndim

    def __call__(self, data: ProteinLigandData):
        """
        Func:
            data.ligand_atom_mask, data.ligand_decomp_mask --> data.ligand_atom_aux_feature
            data.pocket_atom_masks --> concat data.protein_atom_feature
        """
        data.prior_group_idx = torch.LongTensor(list(range(data.num_arms)) + [data.num_arms])
        data.max_decomp_group = self.num_classes

        if self.add_to_ligand:
            # ligand decomp mask: change scaffold index from -1 to data.num_arms, convenient for one-hotting
            data.ligand_decomp_mask = data.ligand_atom_mask.clone()
            data.ligand_decomp_mask[data.ligand_decomp_mask == -1] = data.num_arms
            # add ligand atom features and protein atom features
            arm_ind = F.one_hot((data.ligand_atom_mask >= 0).long(), num_classes=2)
            arm_scaffold_index = F.one_hot(data.ligand_decomp_mask, self.num_classes)
            data.ligand_decomp_group_idx = data.ligand_decomp_mask
            if self.add_ord_feat:
                # Note: aux feature should be independent to pos
                data.ligand_atom_aux_feature = torch.cat([arm_scaffold_index, arm_ind], -1)
            else:
                data.ligand_atom_aux_feature = arm_ind

        if self.add_to_protein:
            protein_arm_ind = F.one_hot((data.pocket_atom_masks.sum(0) > 0).long(), num_classes=2)
            protein_arm_scaffold_index = torch.zeros([len(data.protein_pos), self.num_classes])
            for arm_id, mask in enumerate(data.pocket_atom_masks):
                # special case: no surrounding protein atoms near an arm (data id: 86623)
                if mask.sum() == 0:
                    continue
                protein_arm_scaffold_index[mask][arm_id] = 1
            # dummy idx now
            data.protein_decomp_group_idx = torch.LongTensor([-1] * len(data.protein_pos))
            if self.add_ord_feat:
                data.protein_atom_feature = torch.cat(
                    [data.protein_atom_feature, protein_arm_scaffold_index, protein_arm_ind], -1)
            else:
                data.protein_atom_feature = torch.cat([data.protein_atom_feature, protein_arm_ind], -1)
        return data


class AddDummyAtom(object):
    def __init__(self):
        super().__init__()
        
    def __call__(self, data: ProteinLigandData):
        """
        Func:
            expand following features with dummy atom:
            data.ligand_element, data.ligand_atom_feature, data.ligand_hybridization, data.ligand_pos
            data.arm_num_atoms, data.scaffold_num_atoms, data.ligand_decomp_num_atoms, data.ligand_atom_mask
            data.ligand_fc_bond_type, data.ligand_fc_bond_index (not need to expand?)
        """
        data.ligand_element = torch.cat([data.ligand_element, torch.zeros_like(data.ligand_element)])
        data.ligand_hybridization = torch.cat([data.ligand_hybridization, \
            ['UNSPECIFIED' for i in range(len(data.ligand_hybridization))]])
        data.ligand_atom_feature = torch.cat([data.ligand_atom_feature, torch.zeros_like(data.ligand_atom_feature)], dim=0)
        data.ligand_pos = torch.cat([data.ligand_pos, torch.rand_like(data.ligand_pos)], dim=0)


class FeaturizeLigandBond(object):

    def __init__(self, mode='fc', set_bond_type=False):
        super().__init__()
        self.mode = mode
        self.set_bond_type = set_bond_type

    def __call__(self, data: ProteinLigandData):
        if self.mode == 'fc':
            n_atoms = len(data.ligand_atom_mask)  # only ligand atom mask is reset in beta prior sampling
            full_dst = torch.repeat_interleave(torch.arange(n_atoms), n_atoms)
            full_src = torch.arange(n_atoms).repeat(n_atoms)
            mask = full_dst != full_src
            full_dst, full_src = full_dst[mask], full_src[mask]
            data.ligand_fc_bond_index = torch.stack([full_src, full_dst], dim=0)
            assert data.ligand_fc_bond_index.size(0) == 2
        elif self.mode == 'decomp_fc':
            all_full_src, all_full_dst = [], []
            for i in range(data.num_arms + data.num_scaffold):
                n_atoms = (data.ligand_decomp_mask == i).sum()
                arm_atom_idx = (data.ligand_decomp_mask == i).nonzero()[:, 0]
                full_dst = torch.repeat_interleave(arm_atom_idx, n_atoms)
                full_src = arm_atom_idx.repeat(n_atoms)
                mask = full_dst != full_src
                full_dst, full_src = full_dst[mask], full_src[mask]
                all_full_src.append(full_src)
                all_full_dst.append(full_dst)
            all_full_src = torch.cat(all_full_src)
            all_full_dst = torch.cat(all_full_dst)
            data.ligand_fc_bond_index = torch.stack([all_full_src, all_full_dst], dim=0)
            assert data.ligand_fc_bond_index.size(0) == 2
        elif self.mode == 'scaffold_fc':
            all_full_src, all_full_dst = [], []
            for i in range(data.num_arms):
                n_atoms = (data.ligand_atom_mask == i).sum()
                arm_atom_idx = (data.ligand_decomp_mask == i).nonzero()[:, 0]
                full_dst = torch.repeat_interleave(arm_atom_idx, n_atoms)
                full_src = arm_atom_idx.repeat(n_atoms)
                mask = full_dst != full_src
                full_dst, full_src = full_dst[mask], full_src[mask]
                all_full_src.append(full_src)
                all_full_dst.append(full_dst)

            n_atoms = len(data.ligand_atom_mask)
            sca_atom_idx = (data.ligand_atom_mask == -1).nonzero()[:, 0]
            sca_dst = torch.repeat_interleave(torch.arange(n_atoms), len(sca_atom_idx))
            sca_src = sca_atom_idx.repeat(n_atoms)
            mask = sca_dst != sca_src
            sca_dst, sca_src = sca_dst[mask], sca_src[mask]
            all_full_src.append(sca_src)
            all_full_dst.append(sca_dst)

            all_full_src = torch.cat(all_full_src)
            all_full_dst = torch.cat(all_full_dst)
            data.ligand_fc_bond_index = torch.stack([all_full_src, all_full_dst], dim=0)
            assert data.ligand_fc_bond_index.size(0) == 2

        else:
            raise ValueError(self.mode)

        if hasattr(data, 'ligand_bond_index') and self.set_bond_type:
            n_atoms = data.ligand_pos.size(0)
            bond_matrix = torch.zeros(n_atoms, n_atoms).long()
            src, dst = data.ligand_bond_index
            bond_matrix[src, dst] = data.ligand_bond_type
            data.ligand_fc_bond_type = bond_matrix[data.ligand_fc_bond_index[0], data.ligand_fc_bond_index[1]]
        return data


class RandomRotation(object):

    def __init__(self):
        super().__init__()

    def __call__(self,  data: ProteinLigandData):
        M = np.random.randn(3, 3)
        Q, __ = np.linalg.qr(M)
        Q = torch.from_numpy(Q.astype(np.float32))
        data.ligand_pos = data.ligand_pos @ Q
        data.protein_pos = data.protein_pos @ Q
        return data


class AmberFeaturizer(object):

    def __init__(self, bond_info_path, angle_info_path):
        super().__init__()
        # Amber Energy
        bond_info = torch.load(bond_info_path)
        angle_info = torch.load(angle_info_path)
        for k, v in bond_info.items():
            bond_info[k] = [np.mean(v), np.std(v)]
        for k, v in angle_info.items():
            v = [i * (math.pi / 180) for i in v]
            angle_info[k] = [np.mean(v), np.std(v)]
        
        self.bond_info = bond_info
        self.angle_info = angle_info

    def getAngel(self, posi, posj, posk, eps=1e-16, degree=False):
        rJI = posi - posj
        n_rJI = torch.norm(rJI)
        if n_rJI < eps:
            raise ValueError
        rJK = posk - posj
        n_rJK = torch.norm(rJK)
        if n_rJK < eps:
            raise ValueError
        
        dot_prod = torch.dot(rJI, rJK)
        cos_angle = dot_prod / (n_rJI * n_rJK)
        cos_angle = torch.clamp(cos_angle, -1.0, 1.0)
        angle = torch.acos(cos_angle)
        if degree:
            angle = torch.rad2deg(angle)
        
        return angle    

    def relu_mse_score(self, value, stat_info):
        value = abs(value - stat_info[0]) - 3 * stat_info[1]
        return value**2 if value > 0 else 0

    def amber_energy(self, rdmol):
        conf = rdmol.GetConformer()
        bond_energy = 0
        for bond in rdmol.GetBonds():
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            atom1_type = atom1.GetSymbol()
            atom2_type = atom2.GetSymbol()
            atom1_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(atom1_type)]
            atom2_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(atom2_type)]

            bond_length = rdMolTransforms.GetBondLength(conf, atom1.GetIdx(), atom2.GetIdx())
            if atom1_idx < atom2_idx:
                if (atom1_idx, atom2_idx) in self.bond_info:
                    bond_energy += self.relu_mse_score(bond_length, self.bond_info[(atom1_idx, atom2_idx)])
            else:
                if (atom2_idx, atom1_idx) in self.bond_info:
                    bond_energy += self.relu_mse_score(bond_length, self.bond_info[(atom2_idx, atom1_idx)])

        angle_energy = 0
        for atom in rdmol.GetAtoms():
            nbrs = [nbr for nbr in atom.GetNeighbors()]
            if len(nbrs) > 1:
                atom_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(atom.GetSymbol())]
                for nb1, nb2 in combinations(nbrs, 2):
                    nb1_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(nb1.GetSymbol())]
                    nb2_idx = MAP_ATOM_TYPE_ONLY_TO_INDEX[Chem.GetPeriodicTable().GetAtomicNumber(nb2.GetSymbol())]
                    ang = rdMolTransforms.GetAngleDeg(conf, nb1.GetIdx(), atom.GetIdx(), nb2.GetIdx())
                    rad = ang * (math.pi / 180)
                    if nb1_idx < nb2_idx:
                        if (nb1_idx, atom_idx, nb2_idx) in self.angle_info:
                            angle_energy += self.relu_mse_score(rad, self.angle_info[(nb1_idx, atom_idx, nb2_idx)])
                    else:
                        if (nb2_idx, atom_idx, nb1_idx) in self.angle_info:
                            angle_energy += self.relu_mse_score(rad, self.angle_info[(nb2_idx, atom_idx, nb1_idx)])

        return bond_energy + angle_energy

    def __call__(self,  data: ProteinLigandData):
        rdmol = data.ligand_rdmol
        data.amber_energy = self.amber_energy(rdmol)

        for arm in data.arms_list:
            arm['amber_energy'] = self.amber_energy(arm['arm'])

        for arm in data.scaffold_list:
            if 'arm' in arm:
                arm['amber_energy'] = self.amber_energy(arm['arm'])
        
        return data