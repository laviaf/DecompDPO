from rdkit import Chem
import numpy as np
from rdkit.Chem import rdMolTransforms
from copy import deepcopy
import re
import random
from functools import cmp_to_key
from rdkit.Chem.rdMolAlign import AlignMol

REAL_ATOM_MASK = 4
DUMMY_ATOM_MASK = 2
H_DUMMY_ATOM_MASK = 1


def get_cano_mol(mol):
    cano_smi = Chem.MolToSmiles(mol)
    cano_mol = Chem.MolFromSmiles(cano_smi)
    match = cano_mol.GetSubstructMatch(mol)
    # print(match)
    if not match:
        cano_mol = Chem.MolFromSmarts(cano_smi)
        match = cano_mol.GetSubstructMatch(mol)
        # print(match)
    # print(match)
    return cano_mol


# utils
def get_clean_smi(mol):
    rdmol = get_clean_mol(mol)
    return Chem.MolToSmiles(rdmol)


def get_clean_mol(mol):
    rdmol = deepcopy(mol)
    for at in rdmol.GetAtoms():
        at.SetAtomMapNum(0)
        at.SetIsotope(0)
    Chem.RemoveStereochemistry(rdmol)
    return rdmol


def break_star_symm(mol):
    mol = deepcopy(mol)
    maps1 = list(Chem.CanonicalRankAtoms(mol, breakTies=False, includeChirality=False))
    # maps2 = list(Chem.CanonicalRankAtoms(mol, breakTies=True))
    # print(maps1)
    atoms = mol.GetAtoms()
    counter = {}
    for x in maps1:
        if x not in counter:
            counter[x] = 1
        else:
            counter[x] += 1
    start = 100
    unique = [v for v, freq in counter.items() if freq == 1]
    # print(unique)
    for i, at in enumerate(atoms):
        # print(i,k)
        if (at.GetSymbol() == 'Hg' or at.GetSymbol() == '*') and maps1[i] not in unique:
            at.SetAtomMapNum(start)
            start += 1
    return mol


def get_align_map(mol, cano_mol):
    atom_map1, atom_map2 = {}, {}
    for i, j in enumerate(cano_mol.GetSubstructMatch(mol)):
        atom_map1[i] = j
        atom_map2[j] = i
    for i, atom in enumerate(mol.GetAtoms()):
        if (atom.GetSymbol() == '*' or atom.GetSymbol() == 'Hg') and atom.GetAtomMapNum() != 0:
            atom_map1[i] = atom.GetAtomMapNum()
            atom_map2[atom.GetAtomMapNum()] = i
    return (atom_map1, atom_map2)


def find_parts_bonds(mol, parts):
    ret_bonds = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            i_part = parts[i]
            j_part = parts[j]
            for i_atom_idx in i_part:
                for j_atom_idx in j_part:
                    bond = mol.GetBondBetweenAtoms(i_atom_idx, j_atom_idx)
                    if bond is None:
                        continue
                    ret_bonds.append((i_atom_idx, j_atom_idx))
    return ret_bonds


def get_other_atom_idx(mol, atom_idx_list):
    ret_atom_idx = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() not in atom_idx_list:
            ret_atom_idx.append(atom.GetIdx())
    return ret_atom_idx


def get_rings(mol):
    rings = []
    for ring in list(Chem.GetSymmSSSR(mol)):
        ring = list(ring)
        rings.append(ring)
    return rings


def get_bonds(mol, bond_type):
    bonds = []
    for bond in mol.GetBonds():
        if bond.GetBondType() is bond_type:
            bonds.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    return bonds


def get_center(mol, confId=-1):
    conformer = mol.GetConformer(confId)
    center = np.mean(conformer.GetPositions(), axis=0)
    return center


def trans(x, y, z):
    translation = np.eye(4)
    translation[:3, 3] = [x, y, z]
    return translation


def centralize(mol, confId=-1):
    mol = deepcopy(mol)
    conformer = mol.GetConformer(confId)
    center = get_center(mol, confId)
    translation = trans(-center[0], -center[1], -center[2])
    rdMolTransforms.TransformConformer(conformer, translation)
    return mol


def canonical_frag_smi(frag_smi):
    frag_smi = re.sub(r'\[\d+\*\]', '[*]', frag_smi)
    canonical_frag_smi = Chem.CanonSmiles(frag_smi)
    return canonical_frag_smi


def get_surrogate_frag(frag):
    frag = deepcopy(frag)
    m_frag = Chem.RWMol(frag)
    for atom in m_frag.GetAtoms():
        if atom.GetSymbol() == '*':
            atom_idx = atom.GetIdx()
            m_frag.ReplaceAtom(atom_idx, Chem.Atom(PLACE_HOLDER_ATOM))
    Chem.SanitizeMol(m_frag)
    return m_frag


def get_align_points(frag1, frag2):
    align_point1 = np.zeros((frag1.GetNumAtoms(), 3))
    align_point2 = np.zeros((frag2.GetNumAtoms(), 3))
    frag12frag2 = dict()
    frag22farg1 = dict()
    order1 = list(Chem.CanonicalRankAtoms(frag1, breakTies=True))
    order2 = list(Chem.CanonicalRankAtoms(frag2, breakTies=True))
    con1 = frag1.GetConformer()
    con2 = frag2.GetConformer()
    for i in range(len(order1)):
        frag_idx1 = order1.index(i)
        frag_idx2 = order2.index(i)
        assert frag1.GetAtomWithIdx(frag_idx1).GetSymbol() == frag2.GetAtomWithIdx(frag_idx2).GetSymbol()
        atom_pos1 = list(con1.GetAtomPosition(frag_idx1))
        atom_pos2 = list(con2.GetAtomPosition(frag_idx2))
        align_point1[i] = atom_pos1
        align_point2[i] = atom_pos2
        frag12frag2[frag_idx1] = frag_idx2
        frag22farg1[frag_idx2] = frag_idx1
    return align_point1, align_point2, frag12frag2, frag22farg1


def get_atom_mapping_between_frag_and_surrogate(frag, surro):
    con1 = frag.GetConformer()
    con2 = surro.GetConformer()
    pos2idx1 = dict()
    pos2idx2 = dict()
    for atom in frag.GetAtoms():
        pos2idx1[tuple(con1.GetAtomPosition(atom.GetIdx()))] = atom.GetIdx()
    for atom in surro.GetAtoms():
        pos2idx2[tuple(con2.GetAtomPosition(atom.GetIdx()))] = atom.GetIdx()
    frag2surro = dict()
    surro2frag = dict()
    for key in pos2idx1.keys():
        frag_idx = pos2idx1[key]
        surro_idx = pos2idx2[key]
        frag2surro[frag_idx] = surro_idx
        surro2frag[surro_idx] = frag_idx
    return frag2surro, surro2frag


def get_tree(adj_dict, start_idx, visited, iter_num):
    ret = [start_idx]
    visited.append(start_idx)
    for i in range(iter_num):
        if (not i in visited) and ((start_idx, i) in adj_dict):
            ret.append(get_tree(adj_dict, i, visited, iter_num))
    visited.pop()
    return ret


def get_tree_high(tree):
    if len(tree) == 1:
        return 1

    subtree_highs = []
    for subtree in tree[1:]:
        subtree_high = get_tree_high(subtree)
        subtree_highs.append(subtree_high)

    return 1 + max(subtree_highs)


def tree_sort_cmp(a_tree, b_tree):
    a_tree_high = get_tree_high(a_tree)
    b_tree_high = get_tree_high(b_tree)

    if a_tree_high < b_tree_high:
        return -1
    if a_tree_high > b_tree_high:
        return 1
    return random.choice([-1, 1])


def tree_linearize(tree, res):
    res.append(tree[0])

    subtrees = tree[1:]
    subtrees.sort(key=cmp_to_key(tree_sort_cmp))

    for subtree in subtrees:
        if subtree != subtrees[-1]:
            res.append('b')
            tree_linearize(subtree, res)
            res.append('e')
        else:
            tree_linearize(subtree, res)


# -----
def set_rdmol_positions_(mol, pos):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    assert mol.GetNumAtoms() == pos.shape[0]
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(pos.shape[0]):
        conf.SetAtomPosition(i, pos[i].tolist())
    mol.AddConformer(conf, assignId=True)

    # for i in range(pos.shape[0]):
    #     mol.GetConformer(0).SetAtomPosition(i, pos[i].tolist())
    return mol


def set_rdmol_positions(rdkit_mol, pos, reset=True):
    """
    Args:
        rdkit_mol:  An `rdkit.Chem.rdchem.Mol` object.
        pos: (N_atoms, 3)
    """
    mol = deepcopy(rdkit_mol)
    if reset:
        mol.RemoveAllConformers()
    set_rdmol_positions_(mol, pos)
    return mol


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    if s == 0:
        rotation_matrix = np.eye(3)
    else:
        rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix


def connect_mols(mol1, mol2, atom1_idx, atom2_idx, align=True):
    atom1 = mol1.GetAtomWithIdx(atom1_idx)
    atom2 = mol2.GetAtomWithIdx(atom2_idx)
    neighbor1_idx = atom1.GetNeighbors()[0].GetIdx()
    neighbor2_idx = atom2.GetNeighbors()[0].GetIdx()

    atom1_pos = mol1.GetConformer().GetAtomPosition(atom1_idx)
    atom2_pos = mol2.GetConformer().GetAtomPosition(atom2_idx)
    neighbor1_pos = mol1.GetConformer().GetAtomPosition(neighbor1_idx)
    neighbor2_pos = mol2.GetConformer().GetAtomPosition(neighbor2_idx)

    vec1 = np.array(neighbor1_pos - atom1_pos)
    vec2 = np.array(atom2_pos - neighbor2_pos)

    if align:
        rot_mat = rotation_matrix_from_vectors(vec2, vec1)
        t = atom1_pos - neighbor2_pos  # align neighbor2 with atom1
        aligned_mol2_pos = (mol2.GetConformer().GetPositions() - neighbor2_pos) @ rot_mat.T + atom1_pos
        mol2 = set_rdmol_positions(mol2, aligned_mol2_pos)

    combined = Chem.CombineMols(mol1, mol2)
    emol = Chem.EditableMol(combined)
    bond_order = atom2.GetBonds()[0].GetBondType()
    emol.AddBond(neighbor1_idx,
                 neighbor2_idx + mol1.GetNumAtoms(),
                 order=bond_order)
    emol.RemoveAtom(atom2_idx + mol1.GetNumAtoms())
    emol.RemoveAtom(atom1_idx)
    mol = emol.GetMol()
    Chem.GetSymmSSSR(mol)

    return mol


def get_neighbor(mol, atom_idx, unique=True):
    atom = mol.GetAtomWithIdx(atom_idx)
    n_idx = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != atom_idx]
    if unique and len(n_idx) > 1:
        raise ValueError('unique is set True but multiple neighbors found!')
    return n_idx[0]


def get_twohop_neighbors(mol, atom_idx, global_atom_idx=None):
    atom = mol.GetAtomWithIdx(atom_idx)
    if global_atom_idx is None:
        n1_idx = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != atom_idx][0]
    else:
        n1_idx = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() != atom_idx and global_atom_idx[n.GetIdx()] >= 0][0]

    n1_atom = mol.GetAtomWithIdx(n1_idx)
    if global_atom_idx is None:
        n2_idx = [n.GetIdx() for n in n1_atom.GetNeighbors() if n.GetIdx() not in [atom_idx, n1_idx]][0]
    else:
        n2_idx = [n.GetIdx() for n in n1_atom.GetNeighbors()
                  if n.GetIdx() not in [atom_idx, n1_idx] and global_atom_idx[n.GetIdx()] >= 0][0]
    return (n1_idx, n2_idx)


# get base structure
def replace_atom_in_mol(ori_mol, src_atom, dst_atom):
    mol = deepcopy(ori_mol)
    m_mol = Chem.RWMol(mol)
    for atom in m_mol.GetAtoms():
        if atom.GetAtomicNum() == src_atom:
            atom_idx = atom.GetIdx()
            m_mol.ReplaceAtom(atom_idx, Chem.Atom(dst_atom))
    return m_mol.GetMol()


def get_mol_match(mol, base_mol, method='cano_rank'):
    if method == 'substruct_match':
        # sometimes GetSubstructMatch is problematic, e.g. c1nn[nH]n1
        base_smi = Chem.MolToSmiles(base_mol)
        match = mol.GetSubstructMatch(base_mol)
        if len(match) == 0:
            # try to remove '/'
            fix_base_smi = base_smi.replace('/', '')
            fix_base_smi = fix_base_smi.replace('\\', '')
            base_mol = Chem.MolFromSmiles(fix_base_smi)
            match = mol.GetSubstructMatch(base_mol)
    elif method == 'cano_rank':
        # assume base_mol has already been reordered
        # sometimes it's wrong!  e.g. 'O=C1NC[C@@H]2[C@H]1CN1CCC[C@@H]21'
        mol_no_dummy = replace_atom_in_mol(mol, src_atom=0, dst_atom=1)
        mol_no_dummy = Chem.RemoveAllHs(mol_no_dummy)
        assert mol_no_dummy.GetNumAtoms() == base_mol.GetNumAtoms(), Chem.MolToSmiles(base_mol)
        match = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(mol_no_dummy))])))[1]
    else:
        raise NotImplementedError
    return match, base_mol


def get_best_alignment(ref_mol, probe_mol):
    assert ref_mol.GetNumAtoms() == probe_mol.GetNumAtoms()
    best_rms = 1000
    best_match = None
    for match in probe_mol.GetSubstructMatches(ref_mol, uniquify=False):
        rms = AlignMol(ref_mol, probe_mol, atomMap=[(i, match[i]) for i in range(ref_mol.GetNumAtoms())])
        if rms < best_rms:
            best_rms = rms
            best_match = match
    # This transform is then applied to the specified conformation in the probe molecule
    best_rms = AlignMol(ref_mol, probe_mol, atomMap=[(i, best_match[i]) for i in range(ref_mol.GetNumAtoms())])
    return best_rms, best_match


# def fill_hydrogen_for_matching(mol, base_mol, cano_attach_atoms, align=False):
#     # todo: move base_mol and align out
#     # match = mol.GetSubstructMatch(base_mol)
#     if align:
#         match, _ = get_mol_match(mol, base_mol, method='cano_rank')
#         assert len(match) > 0
#         fill_mol = Chem.AddHs(mol, addCoords=True)
#         fill_mol = Chem.RWMol(fill_mol)
#         all_attach_idx = {match[c_idx]: v for c_idx, v in cano_attach_atoms.items()}
#     else:
#         fill_mol = Chem.AddHs(mol, addCoords=True)
#         fill_mol = Chem.RWMol(fill_mol)
#         all_attach_idx = deepcopy(cano_attach_atoms)
#
#     # minus current num dummys
#     for bond in fill_mol.GetBonds():
#         start = bond.GetBeginAtomIdx()
#         end = bond.GetEndAtomIdx()
#         if start in all_attach_idx and fill_mol.GetAtomWithIdx(end).GetSymbol() == '*':
#             all_attach_idx[start] -= 1
#         elif end in all_attach_idx and fill_mol.GetAtomWithIdx(start).GetSymbol() == '*':
#             all_attach_idx[end] -= 1
#
#     h_dummys = []
#     # fill new dummys
#     for bond in fill_mol.GetBonds():
#         start = bond.GetBeginAtomIdx()
#         end = bond.GetEndAtomIdx()
#         if start in all_attach_idx and fill_mol.GetAtomWithIdx(end).GetSymbol() == 'H' and \
#                 all_attach_idx[start] > 0:
#             all_attach_idx[start] -= 1
#             fill_mol.ReplaceAtom(end, Chem.Atom(0))
#             h_dummys.append(end)
#         elif end in all_attach_idx and fill_mol.GetAtomWithIdx(start).GetSymbol() == 'H' and \
#                 all_attach_idx[end] > 0:
#             all_attach_idx[end] -= 1
#             fill_mol.ReplaceAtom(start, Chem.Atom(0))
#             h_dummys.append(start)
#     new_mol = fill_mol.GetMol()
#     new_mol = Chem.RemoveHs(new_mol)
#     # print('final attach idx: ', all_attach_idx)
#     assert all([v == 0 for v in all_attach_idx.values()]), Chem.MolToSmiles(base_mol)
#     return new_mol, h_dummys


def fill_hydrogen_for_matching(mol, base_mol, cano_attach_atoms, align=False):
    # todo: move base_mol and align out
    # match = mol.GetSubstructMatch(base_mol)
    if align:
        match, _ = get_mol_match(mol, base_mol, method='cano_rank')
        assert len(match) > 0
        fill_mol = Chem.AddHs(mol, addCoords=True)
        fill_mol = Chem.RWMol(fill_mol)
        # all_attach_idx = {match[c_idx]: v for c_idx, v in cano_attach_atoms.items()}
        all_attach_idx = [match[c_idx] for c_idx in cano_attach_atoms]
    else:
        fill_mol = Chem.AddHs(mol, addCoords=True)
        fill_mol = Chem.RWMol(fill_mol)
        all_attach_idx = deepcopy(cano_attach_atoms)

    # minus current num dummys
    # for bond in fill_mol.GetBonds():
    #     start = bond.GetBeginAtomIdx()
    #     end = bond.GetEndAtomIdx()
    #     if start in all_attach_idx and fill_mol.GetAtomWithIdx(end).GetSymbol() == '*':
    #         all_attach_idx[start] -= 1
    #     elif end in all_attach_idx and fill_mol.GetAtomWithIdx(start).GetSymbol() == '*':
    #         all_attach_idx[end] -= 1

    # h_dummys = []
    # fill new dummys
    for bond in fill_mol.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if start in all_attach_idx and fill_mol.GetAtomWithIdx(end).GetSymbol() == 'H':
            fill_mol.ReplaceAtom(end, Chem.Atom(0))
            # h_dummys.append(end)
        elif end in all_attach_idx and fill_mol.GetAtomWithIdx(start).GetSymbol() == 'H':
            fill_mol.ReplaceAtom(start, Chem.Atom(0))
            # h_dummys.append(start)
    new_mol = fill_mol.GetMol()
    new_mol = Chem.RemoveHs(new_mol)
    # print('final attach idx: ', all_attach_idx)
    # assert all([v == 0 for v in all_attach_idx.values()]), Chem.MolToSmiles(base_mol)
    return new_mol

def get_frag_by_atom(mol, atom_idx):
    # if kekulize:
    m = deepcopy(mol)
    Chem.Kekulize(m, clearAromaticFlags=True)
    # m =  Chem.MolFromSmiles(Chem.MolFragmentToSmiles(m, atom_idx, kekuleSmiles=True))
    emol = Chem.RWMol()
    id_map = {}
    for i,a_id in enumerate(atom_idx):
        emol.AddAtom(m.GetAtomWithIdx(int(a_id)))
        id_map[a_id] = i
        
    for bond in m.GetBonds():
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        if start in atom_idx and end in atom_idx:
            emol.AddBond(id_map[start],id_map[end],bond.GetBondType())
    m = emol.GetMol()
    # else:
    #     m =  Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atom_idx, kekuleSmiles=True))
    pos = mol.GetConformer().GetPositions()
    m = set_rdmol_positions(m,pos[atom_idx])
    return m