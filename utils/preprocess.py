import copy
import itertools
import re

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS, AllChem
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
from utils.fragmentation.utils import get_frag_by_atom
from utils.misc import DecomposeError

INF = 999999

def decompose_molecule(mol, method='BRICS'):
    mol = copy.deepcopy(mol)
    if method == 'BRICS':
        # get bonds need to be break
        bonds = [bond[0] for bond in list(BRICS.FindBRICSBonds(mol))]
        
        # whether the molecule can really be break
        if len(bonds) != 0:
            bond_ids = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

            # break the bonds & set the dummy labels for the bonds
            dummyLabels = [(0, 0) for i in range(len(bond_ids))]
            break_mol = Chem.FragmentOnBonds(mol, bond_ids, dummyLabels=dummyLabels)
            du = Chem.MolFromSmiles('*')
            break_mol_wodu = AllChem.ReplaceSubstructs(break_mol,du,Chem.MolFromSmiles('[H]'),True)[0]
            break_mol_wodu = Chem.RemoveHs(break_mol_wodu)

            atom_map = []
            frags = list(Chem.GetMolFrags(break_mol_wodu, asMols=True, fragsMolAtomMapping=atom_map))
            smi = [Chem.MolToSmiles(f) for f in frags]
            return smi, [[list(m)] for m in atom_map] # for align with previous version
        else:
            return Chem.MolToSmiles(mol), [[i for i in range(mol.GetNumAtoms())]]

    else:
        raise NotImplementedError


def find_complete_seg(current_idx_set, current_match_list, all_atom_idx, num_element):
    if len(all_atom_idx) == 0:
        if len(current_idx_set) == num_element:
            return current_match_list
        else:
            return None

    raw_matches = all_atom_idx[0]
    all_matches_subset = []
    # trim the matches list
    matches = []
    for match in raw_matches:
        if any([x in current_idx_set for x in match]):
            continue
        matches.append(match)

    for L in reversed(range(1, min(len(matches) + 1, num_element - len(current_idx_set) + 1))):
        for subset in itertools.combinations(matches, L):
            subset = list(itertools.chain(*subset))
            if len(subset) == len(set(subset)) and \
                    len(set(subset + list(current_idx_set))) == len(subset) + len(current_idx_set):
                all_matches_subset.append(subset)
    # print('current idx set: ', current_idx_set, 'all match subset: ', all_matches_subset)

    for match in all_matches_subset:
        valid = True
        for i in match:
            if i in current_idx_set:
                valid = False
                break
        if valid:
            next_idx_set = copy.deepcopy(current_idx_set)
            next_match_list = copy.deepcopy(current_match_list)
            for i in match:
                next_idx_set.add(i)
            next_match_list.append(match)

            match_list = find_complete_seg(next_idx_set, next_match_list, all_atom_idx[1:], num_element)
            if match_list is not None:
                return match_list


def compute_pocket_frag_distance(pocket_centers, frag_centroid):
    all_distances = []
    for center in pocket_centers:
        distance = np.linalg.norm(frag_centroid - center, ord=2)
        all_distances.append(distance)
    return np.mean(all_distances)


def is_terminal_frag(mol, frag_atom_idx):
    split_bond_idx = []
    for bond_idx, bond in enumerate(mol.GetBonds()):
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if (start in frag_atom_idx) != (end in frag_atom_idx):
            split_bond_idx.append(bond_idx)
    return len(split_bond_idx) <= 1  # equals 0 when only one fragment is detected


def get_submol(mol, split_bond_idx, pocket_atom_idx):
    if len(pocket_atom_idx) == 0:
        return None
    elif len(pocket_atom_idx) == mol.GetNumAtoms() and len(split_bond_idx) == 0:
        return copy.deepcopy(mol)
    else:
        r = Chem.FragmentOnBonds(mol, split_bond_idx)
        frags = Chem.GetMolFrags(r)
        frags_overlap_atoms = [len(set(pocket_atom_idx).intersection(set(frag))) for frag in frags]
        hit_idx = np.argmax(frags_overlap_atoms)
        submol = Chem.GetMolFrags(r, asMols=True)[hit_idx]
        return submol


def extract_submols(mol, pocket_list, debug=False, verbose=False):
    # decompose molecules into fragments
    try:
        union_frags_smiles, possible_frags_atom_idx = decompose_molecule(mol)
    except:
        raise DecomposeError
    # each element is a group of fragments with the same type
    match_frags_list = find_complete_seg(set(), [], possible_frags_atom_idx, mol.GetNumAtoms())
    if match_frags_list is None:
        raise DecomposeError
    # flatten the matching list
    frags_smiles_list, frags_atom_idx_list = [], []
    for smiles, group_atom_idx in zip(union_frags_smiles, match_frags_list):
        query_frag_mol = Chem.MolFromSmiles(smiles)
        if len(group_atom_idx) == query_frag_mol.GetNumAtoms():
            frags_smiles_list.append(smiles)
            frags_atom_idx_list.append(group_atom_idx)
        else:
            assert len(group_atom_idx) % query_frag_mol.GetNumAtoms() == 0
            n_atoms = 0
            for match in mol.GetSubstructMatches(query_frag_mol):
                if all([atom_idx in group_atom_idx for atom_idx in match]):
                    frags_smiles_list.append(smiles)
                    frags_atom_idx_list.append([atom_idx for atom_idx in match])
                    n_atoms += len(match)
            assert n_atoms == len(group_atom_idx)
    # find centroid of each fragment
    ligand_pos = mol.GetConformer().GetPositions()
    dist_mat = np.zeros([len(frags_smiles_list), len(pocket_list)])

    all_frag_centroid = []
    for frag_idx, (frag_smiles, frag_atom_idx) in enumerate(zip(frags_smiles_list, frags_atom_idx_list)):
        frag_pos = np.array([ligand_pos[atom_idx] for atom_idx in frag_atom_idx])
        frag_centroid = np.mean(frag_pos, 0)
        all_frag_centroid.append(frag_centroid)

        for pocket_idx, pocket in enumerate(pocket_list):
            centers = [a.centroid for a in pocket.alphas]
            distance = compute_pocket_frag_distance(centers, frag_centroid)
            dist_mat[frag_idx, pocket_idx] = distance
    all_frag_centroid = np.array(all_frag_centroid)

    # clustering
    # number of clustering centers: number of pockets (arms) + 1 (scaffold)
    # 1. determine clustering centers
    terminal_mask = np.array([is_terminal_frag(mol, v) for v in frags_atom_idx_list])
    t_frag_idx = (terminal_mask == 1).nonzero()[0]
    nt_frag_idx = (terminal_mask == 0).nonzero()[0]

    pocket_idx, frag_idx = linear_sum_assignment(dist_mat[t_frag_idx].T)
    arms_frag_idx = np.array([t_frag_idx[idx] for idx in frag_idx])
    clustering_centers = [all_frag_centroid[idx] for idx in arms_frag_idx]
    # linear_sum_assignment will handle the case where the amount of arms is greater than the number of pockets
    # if the number of arms is less than the number of pockets, supplement centroid of pocket's alpha atoms
    cluster_pocket_idx = list(pocket_idx)
    if len(clustering_centers) < len(pocket_list):
        if verbose:
            print('warning: less arms than pockets')
        add_pocket_idx = list(set(range(len(pocket_list))) - set(pocket_idx))
        for p_idx in add_pocket_idx:
            centers = [a.centroid for a in pocket_list[p_idx].alphas]
            pocket_centroid = np.mean(centers, 0)
            clustering_centers.append(pocket_centroid)
            cluster_pocket_idx.append(p_idx)
    assert len(clustering_centers) == len(pocket_list)

    # select the frag centroid which is farthest to all existing centers as the scaffold clustering center
    # it is possible that only arm fragments are detected
    non_arm_frag_idx = np.array([idx for idx in range(len(all_frag_centroid)) if idx not in arms_frag_idx])
    if len(non_arm_frag_idx) > 0:
        scaffold_frag_idx = non_arm_frag_idx[
            np.argmax(distance_matrix(all_frag_centroid[non_arm_frag_idx], clustering_centers).sum(-1))]
        clustering_centers.append(all_frag_centroid[scaffold_frag_idx])
    else:
        scaffold_frag_idx = np.array([], dtype=np.int)

    if debug:
        print(f't frag idx: {t_frag_idx} nt frag idx: {nt_frag_idx} '
              f'arms frag idx: {arms_frag_idx} non arm frag idx: {non_arm_frag_idx}')

    # 2. determine assignment
    # todo: can be improved, like updating clustering center
    frag_cluster_dist_mat = distance_matrix(all_frag_centroid, clustering_centers)
    assignment = -1 * np.ones(len(all_frag_centroid)).astype(np.int64)
    assignment[arms_frag_idx] = pocket_idx

    # todo: order problem
    # assignment[scaffold_frag_idx] = len(clustering_centers) - 1
    # for idx in range(len(all_frag_centroid)):
    #     assign_cluster_idx = frag_cluster_dist_mat[idx].argmin()
    #     if assign_cluster_idx == len(clustering_centers) - 1:
    #         # directly assign scaffold
    #         assignment[idx] = len(clustering_centers) - 1
    #     else:
    #         assign_pocket_idx = cluster_pocket_idx[assign_cluster_idx]
    #         # arms --> check validity
    #         current_atom_idx = []
    #         for assign_frag_idx in (assignment == assign_pocket_idx).nonzero()[0]:
    #             current_atom_idx += frags_atom_idx_list[assign_frag_idx]
    #         current_atom_idx += frags_atom_idx_list[idx]

    #         if is_terminal_frag(mol, current_atom_idx):
    #             assignment[idx] = assign_pocket_idx
    #         else:
    #             assignment[idx] = len(clustering_centers) - 1

    assignment[scaffold_frag_idx] = len(clustering_centers) - 1
    loop = 0
    while any(assignment == -1):
        loop += 1
        if loop > 20:
            disconnect_idx = (assignment == -1).nonzero()
            assignment[disconnect_idx] = len(clustering_centers) - 1
            break
        for idx in range(len(all_frag_centroid)):
            if assignment[idx] == -1:
                assign_cluster_idx = frag_cluster_dist_mat[idx].argmin()
                if assign_cluster_idx == len(clustering_centers) - 1:
                    # directly assign scaffold
                    assignment[idx] = len(clustering_centers) - 1
                else:
                    assign_pocket_idx = cluster_pocket_idx[assign_cluster_idx]
                    # arms --> check validity
                    current_atom_idx = []
                    for assign_frag_idx in (assignment == assign_pocket_idx).nonzero()[0]:
                        current_atom_idx += frags_atom_idx_list[assign_frag_idx]
                    current_atom_idx += frags_atom_idx_list[idx]

                    if is_terminal_frag(mol, current_atom_idx):
                        assignment[idx] = assign_pocket_idx
                    else:
                        continue

    # 3. construct submols given assignment
    all_submols = []
    scaffold_bond_idx = []
    all_arm_atom_idx = []
    valid_pocket_id = []
    for pocket_id in range(len(pocket_list)):
        arm_atom_idx = []
        for assigned_idx in (assignment == pocket_id).nonzero()[0]:
            arm_atom_idx += frags_atom_idx_list[assigned_idx]

        split_bond_idx = []
        for bond_idx, bond in enumerate(mol.GetBonds()):
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            if (start in arm_atom_idx) != (end in arm_atom_idx):
                split_bond_idx.append(bond_idx)
        assert len(split_bond_idx) <= 1, split_bond_idx
        scaffold_bond_idx += split_bond_idx

        match_submol = get_submol(mol, split_bond_idx, arm_atom_idx)
        if len(arm_atom_idx) > 0:
            valid_pocket_id.append(pocket_id)
            assert match_submol is not None
            all_arm_atom_idx.append(arm_atom_idx)
            all_submols.append(match_submol)

    scaffold_atom_idx = []
    for assigned_idx in (assignment == len(pocket_list)).nonzero()[0]:
        scaffold_atom_idx += frags_atom_idx_list[assigned_idx]

    scaffold_submol = get_submol(mol, scaffold_bond_idx, scaffold_atom_idx)
    all_submols.append(scaffold_submol)
    flat_arm_atom_idx = list(itertools.chain(*all_arm_atom_idx))
    assert len(flat_arm_atom_idx + scaffold_atom_idx) == len(set(flat_arm_atom_idx + scaffold_atom_idx))
    assert set(flat_arm_atom_idx + scaffold_atom_idx) == set(range(mol.GetNumAtoms()))
    all_submol_atom_idx = all_arm_atom_idx + [scaffold_atom_idx]
    return all_frag_centroid, assignment, all_submol_atom_idx, all_submols, valid_pocket_id


def extract_subpockets(protein, pocket, method, **kwargs):
    if method == 'v1':
        # Method 1: union of lining atom idx / lining residue idx
        pocket_lining_atoms = [atom for atom in kwargs['mdtraj_protein'].atom_slice(pocket.lining_atoms_idx).top.atoms]
        pocket_atom_serial = [atom.serial for atom in pocket_lining_atoms]
        # pocket_res_idx = [atom.residue.resSeq for atom in pocket_lining_atoms]

        selected_atom_serial, selected_residues = [], []
        sel_idx = set()
        for atom in protein.atoms:
            if atom['atom_id'] in pocket_atom_serial:
                chain_res_id = '%s_%s_%d_%s' % (atom['chain'], atom['segment'], atom['res_id'], atom['res_insert_id'])
                sel_idx.add(chain_res_id)

        for res in protein.residues:
            if res['chain_res_id'] in sel_idx:
                selected_residues.append(res)
                selected_atom_serial += [protein.atoms[a_idx]['atom_id'] for a_idx in res['atoms']]

    elif method == 'v2':
        # Method 2: alpha atom --> sphere with a large radius
        centers = [a.centroid for a in pocket.alphas]
        selected_atom_serial, selected_residues = protein.query_residues_centers(
            centers, radius=kwargs['protein_radius'])

    elif method == 'v3':
        # Method 3: atom-level query but select the whole residue
        centers = [a.centroid for a in pocket.alphas]
        selected_atom_serial, selected_residues = protein.query_residues_atom_centers(
            centers, radius=kwargs['protein_radius'])

    elif method == 'submol_radius':
        centers = kwargs['submol'].GetConformer(0).GetPositions()
        selected_atom_serial, selected_residues = protein.query_residues_centers(
            centers, radius=kwargs['protein_radius'])

    else:
        raise NotImplementedError

    return selected_atom_serial, selected_residues


def union_pocket_residues(all_pocket_residues):
    selected = []
    sel_idx = set()
    for pocket_r in all_pocket_residues:
        for r in pocket_r:
            if r['chain_res_id'] not in sel_idx:
                selected.append(r)
                sel_idx.add(r['chain_res_id'])
    return selected


def mark_in_range(query_points, ref_points, cutoff=1.6):
    indices = np.where(distance_matrix(query_points, ref_points) <= cutoff)[0]
    indices = np.unique(indices)
    query_bool = np.zeros(len(query_points), dtype=bool)
    query_bool[indices] = 1
    return query_bool


def split_arms_scaffold(mol, atom_map_id, target_cluster_num):
    '''
    mol, fragments atom index -> arms atom index, scaffold atom index, arms num, scaffold num
    if fragment num > target_cluster_num: merge cloest fragments and call this function again
    else: return temp assignment of arms and scaffold
    '''
    assert target_cluster_num >= 1
    # determine cluster centers
    fragment_centers = []
    terminal_clustering_num = 0
    terminal_centers = []
    non_terminal_centers = []
    terminal_ids = []
    non_terminal_ids = []
    for ids in atom_map_id:
        if is_terminal_frag(mol, ids):
            terminal_clustering_num += 1
            terminal_ids.append(ids)
            frag_pos = get_frag_by_atom(mol, ids).GetConformer(0).GetPositions().mean(0)
            terminal_centers.append(frag_pos)
            fragment_centers.append(frag_pos)
        else:
            non_terminal_ids.append(ids)
            frag_pos = get_frag_by_atom(mol, ids).GetConformer(0).GetPositions().mean(0)
            non_terminal_centers.append(frag_pos)
            fragment_centers.append(frag_pos)

    if len(non_terminal_ids) != 0:
        scaffold_center = [np.mean(non_terminal_centers, axis=0)]
        assign_scaffold_frag = distance_matrix(non_terminal_centers, scaffold_center).argmin()
        scaffold_id = [non_terminal_ids[assign_scaffold_frag]] # assign closest fragment as scaffold frag
        scaffold_center = [non_terminal_centers[assign_scaffold_frag]]
        scaffold_num = 1
    else:
        scaffold_center = []
        scaffold_id = []
        scaffold_num = 0

    # split arms and scaffold
    if target_cluster_num >= len(atom_map_id):
        if len(non_terminal_ids) > 1:
            non_terminal_ids = [np.concatenate(non_terminal_ids).tolist()]
        return terminal_ids, non_terminal_ids, terminal_clustering_num, scaffold_num

    elif target_cluster_num < len(atom_map_id):
        # calculate distance matrixs, merge closest cluster pairs
        terminal_centers.extend(scaffold_center)
        terminal_ids.extend(scaffold_id)
        fragment_mapping = distance_matrix(fragment_centers, terminal_centers)
        fragment_mapping[np.where(fragment_mapping == 0)] = INF

        mid = np.array(np.where(fragment_mapping == fragment_mapping.min()))[:,0]
        atom_map_id.append(np.concatenate([atom_map_id[mid[0]], terminal_ids[mid[1]]]).tolist())
        atom_map_id.remove(atom_map_id[mid[0]])
        atom_map_id.remove(terminal_ids[mid[1]])
        return split_arms_scaffold(mol, atom_map_id, target_cluster_num)