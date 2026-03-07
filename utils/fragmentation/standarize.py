from rdkit import Chem
import numpy as np
from copy import deepcopy
from rdkit.Chem.rdMolAlign import GetBestRMS
from utils.geometry import GetDihedral, SetDihedral, GetDihedralFromPointCloud
from utils.data import process_from_mol
from utils.fragmentation.vocab_gen import FragmentVocab
from utils.fragmentation.utils import get_clean_mol, replace_atom_in_mol, fill_hydrogen_for_matching, \
    get_best_alignment, get_twohop_neighbors, connect_mols, get_neighbor, get_mol_match


def get_frag_3d_idx(frag, g_all_atom_idx, vocab: FragmentVocab, fix_ref_frag_idx=None):
    query_frag = get_clean_mol(frag)
    base_frag = replace_atom_in_mol(query_frag, src_atom=0, dst_atom=1)
    base_frag = Chem.RemoveHs(base_frag)
    base_smi = Chem.MolToSmiles(base_frag)
    v = vocab.get_frags(base_smi)

    if fix_ref_frag_idx is None:
        query_frag = fill_hydrogen_for_matching(query_frag, v.base_mol, v.attach_atom_list, align=True)
        best_idx = None
        best_ref_idx = None
        best_rmsd = 1000

        # search matched fragment type and geometry
        for idx, ref_idx, ref_frag in zip(vocab.get_frag_idx_list(base_smi), range(len(v.rep_mols)), v.rep_mols):
            # ref fragment is aligned
            rmsd, _ = get_best_alignment(ref_frag, query_frag)
            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_idx = idx
                best_ref_idx = ref_idx
        # todo: NoneType best ref idx (data id 114)
        match_ref_frag = deepcopy(v.rep_mols[best_ref_idx])
        assert best_idx is not None
    else:
        best_idx = fix_ref_frag_idx
        best_ref_idx = vocab.get_frag_idx_list(base_smi).index(best_idx)
        match_ref_frag = vocab.get_frag_with_idx(best_idx)['rep_mol']

    # convert local fragment atom idx to global atom idx
    best_rmsd, r2q_mapping = get_best_alignment(match_ref_frag, query_frag)
    aligned_query_frag = Chem.RenumberAtoms(query_frag, r2q_mapping)
    g_all_atom_idx = [g_all_atom_idx[i] if i < len(g_all_atom_idx) else -1 for i in r2q_mapping]

    dummy_mask, cur_dummy = [], 1
    torsion_label = []
    torsion_atom_idx_in_ref = []
    torsion_dummy_idx_in_ref = []
    h_dummy_idx = []
    real_atom_idx = []
    for atom_idx in range(match_ref_frag.GetNumAtoms()):
        atom = match_ref_frag.GetAtomWithIdx(atom_idx)
        if atom.GetSymbol() == '*':
            if r2q_mapping[atom_idx] >= frag.GetNumAtoms():
                h_dummy_idx.append(atom_idx)
                continue
            q_at = frag.GetAtomWithIdx(r2q_mapping[atom_idx])  # frag (not query frag) include dummy isotope info!
            isotope = q_at.GetIsotope()
            if isotope == 0:
                h_dummy_idx.append(atom_idx)
                continue
            torsion_dummy_idx_in_ref.append(atom_idx)
            torsion_label.append(q_at.GetIsotope())
            if match_ref_frag.GetNumAtoms() > 2:
                # should first attempt atom with positive global index (non-dummy atom)
                n1_idx, n2_idx = get_twohop_neighbors(match_ref_frag, atom_idx, g_all_atom_idx)
                torsion_atom_idx_in_ref.append([n1_idx, n2_idx])
            else:
                torsion_atom_idx_in_ref.append([-1, -1])
            cur_dummy += 1
        else:
            real_atom_idx.append(atom_idx)

    frag_info = {
        'match_ref_frag': match_ref_frag,
        'aligned_query_frag': aligned_query_frag,
        'match_base_smiles': base_smi,
        'rmsd': best_rmsd,
        'frag_type': best_idx,
        'num_atoms': frag.GetNumAtoms(),

        'h_dummy_idx': h_dummy_idx,
        'unique_dummy_dict': v.unique_dummy_ids_list[best_ref_idx],
        'unique_dummy_mapping':  {v: k for k, all_v in v.unique_dummy_ids_list[best_ref_idx].items() for v in all_v},
        'dummy_label': torsion_label,
        'torsion_dummy_idx': torsion_dummy_idx_in_ref,
        'torsion_2atom_idx': torsion_atom_idx_in_ref,

        'g_all_atom_idx': g_all_atom_idx,
        'g_atom_idx': [g_all_atom_idx[i] for i in real_atom_idx],
        'g_h_dummy_idx': [g_all_atom_idx[i] for i in h_dummy_idx],
        'g_torsion_dummy_idx': [g_all_atom_idx[i] for i in torsion_dummy_idx_in_ref],
    }
    return frag_info


def add_frag_to_mol(mol, new_frag, mol_atom_idx, frag_atom_idx,
                    mol_torsion_2atom_idx=None, frag_torsion_2atom_idx=None, torsion_angle=None, align=True):
    vocab_mol = deepcopy(new_frag)
    new_mol = connect_mols(mol, vocab_mol, mol_atom_idx, frag_atom_idx, align)

    if mol_torsion_2atom_idx is not None and frag_torsion_2atom_idx is not None:
        first2_torsion_2atom_idx = [idx if idx < mol_atom_idx else idx - 1 for idx in mol_torsion_2atom_idx]
        last2_torsion_2atom_idx = [idx + mol.GetNumAtoms() - 1 if idx < frag_atom_idx else idx + mol.GetNumAtoms() - 2
                                   for idx in frag_torsion_2atom_idx]
        torsion_4atom_idx = first2_torsion_2atom_idx + last2_torsion_2atom_idx
        if torsion_angle is not None:
            SetDihedral(new_mol.GetConformer(0), torsion_4atom_idx, torsion_angle)
        return new_mol, torsion_4atom_idx
    else:
        return new_mol


def get_frags_atom_idx(mol, bond_list):
    # original atom idx in the whole molecule
    src_frags_w_idx = Chem.FragmentOnBonds(mol, bond_list)
    frags_w_idx = list(Chem.GetMolFrags(src_frags_w_idx, asMols=True))  # dummy atom has the true original index
    frags_ori_idx = Chem.GetMolFrags(src_frags_w_idx)  # non-dummy atom has the true original index
    frags_atom_idx = []
    for f_idx, f in enumerate(frags_w_idx):
        r = []
        for atom in f.GetAtoms():
            if atom.GetSymbol() == '*':
                tmp_atom_idx = frags_ori_idx[f_idx][atom.GetIdx()]
                if tmp_atom_idx >= mol.GetNumAtoms():
                    r.append(atom.GetIsotope())
                else:
                    r.append(tmp_atom_idx)
                # r.append(-1)
            else:
                atom_idx = frags_ori_idx[f_idx][atom.GetIdx()]
                r.append(atom_idx)
        frags_atom_idx.append(r)
    return frags_atom_idx


def reorder_frags(frags, dummy_end):
    rot_bond_frag = {i: [] for i in range(1, dummy_end + 1)}
    for frag_idx, frag in enumerate(frags):
        for atom_idx, atom in enumerate(frag.GetAtoms()):
            if atom.GetIsotope() > 0:
                assert atom.GetSymbol() == '*'
                isotope = atom.GetIsotope()
                rot_bond_frag[isotope].append(frag_idx)
    rot_bond_edge_index = np.stack(list(rot_bond_frag.values()), 0)
    remap = {rot_bond_edge_index[0][0]: 0, rot_bond_edge_index[0][1]: 1}
    next_value = 2
    while len(remap) < len(frags):
        for src, dst in rot_bond_edge_index[1:]:
            if (src in remap) != (dst in remap):
                # assert (src in remap) != (dst in remap)
                if src in remap:
                    remap[dst] = next_value
                else:
                    remap[src] = next_value
                next_value += 1
                break
    remap = {v: k for k, v in remap.items()}
    return remap


def extract_all_frag_info(mol, sg, vocab,
                          cut_bond_list=None, fix_frag_idx=None, reorder=True):
    # fragmentize molecule
    src_frags, bond_list, dummy_end = sg.fragmentize(mol, bond_list=cut_bond_list)
    frags = list(Chem.GetMolFrags(src_frags, asMols=True))
    if len(bond_list) == 0:
        frags_atom_idx = [list(range(mol.GetNumAtoms()))]
    else:
        frags_atom_idx = get_frags_atom_idx(mol, bond_list)

    # reorder frags
    if reorder and len(frags) > 1:
        remap = reorder_frags(frags, dummy_end)
        frags = [frags[remap[i]] for i in range(len(frags))]
        frags_atom_idx = [frags_atom_idx[remap[i]] for i in range(len(frags))]

    if fix_frag_idx is None:
        all_frag_info = [get_frag_3d_idx(f, g_atom_idx, vocab) for f, g_atom_idx in zip(frags, frags_atom_idx)]
    else:
        all_frag_info = [get_frag_3d_idx(f, g_atom_idx, vocab, fix_ref_frag_idx=ref_frag_idx)
                         for f, g_atom_idx, ref_frag_idx in zip(frags, frags_atom_idx, fix_frag_idx)]
    return all_frag_info, frags, dummy_end, frags_atom_idx


def build_frag_graph(mol, frags, all_frag_info, dummy_end):
    # compute torsion 4atoms and torsion angles, build fragment-level graph
    rot_bond_frag = {i: [] for i in range(1, dummy_end + 1)}
    rot_bond_attach_atom_idx = {i: [] for i in range(1, dummy_end + 1)}
    rot_bond_torsion_4atoms = {i: [] for i in range(1, dummy_end + 1)}
    g_rot_bond_torsion_4atoms = {i: [] for i in range(1, dummy_end + 1)}
    for frag_idx, frag in enumerate(frags):
        frag_info = all_frag_info[frag_idx]
        for atom_idx, atom in enumerate(frag.GetAtoms()):
            if atom.GetIsotope() > 0:
                assert atom.GetSymbol() == '*'
                isotope = atom.GetIsotope()
                rot_bond_frag[isotope].append(frag_idx)
                index = frag_info['dummy_label'].index(isotope)
                rot_bond_attach_atom_idx[isotope].append(frag_info['torsion_dummy_idx'][index])
                # the first two atoms should be reversed
                if len(rot_bond_torsion_4atoms[isotope]) == 0:
                    l_torsion_atom_idx = list(reversed(frag_info['torsion_2atom_idx'][index]))
                    g_torsion_atom_idx = [frag_info['g_all_atom_idx'][idx] for idx in l_torsion_atom_idx]
                else:
                    l_torsion_atom_idx = frag_info['torsion_2atom_idx'][index]
                    g_torsion_atom_idx = [frag_info['g_all_atom_idx'][idx] for idx in l_torsion_atom_idx]
                rot_bond_torsion_4atoms[isotope] += l_torsion_atom_idx
                g_rot_bond_torsion_4atoms[isotope] += g_torsion_atom_idx

    # print('rot bond frag: ', rot_bond_frag)
    # print('rot bond torsion 4atoms: ', rot_bond_torsion_4atoms)
    # print('g rot bond torsion 4atoms: ', g_rot_bond_torsion_4atoms)
    # gather following the autoregressive order (mimic generation process)
    src_frag_idx, dst_frag_idx = [], []
    src_l_atom_idx, dst_l_atom_idx = [], []
    torsion_4atoms = []
    torsion_angles = []
    ar_torsion_angles, ar_torsion_4atoms = [], []
    conf = mol.GetConformer(0)

    # determine the autoregressive order
    rot_bond_edge_index = np.stack(list(rot_bond_frag.values()), -1)
    valid_mask = np.ones([len(rot_bond_frag)], dtype=bool)
    ar_order = []
    for i in range(1, len(all_frag_info)):
        edge_idx = int(((rot_bond_edge_index <= i).all(0) & valid_mask).nonzero()[0])
        valid_mask[edge_idx] = False
        ar_order.append(edge_idx)

    for k in np.array(ar_order) + 1:
        start, end = rot_bond_frag[k]
        src_frag_idx += [start, end]
        dst_frag_idx += [end, start]

        start_atom, end_atom = rot_bond_attach_atom_idx[k]
        src_l_atom_idx += [start_atom, end_atom]
        dst_l_atom_idx += [end_atom, start_atom]

        r_atoms = g_rot_bond_torsion_4atoms[k]
        ar_torsion_4atoms.append(rot_bond_torsion_4atoms[k])
        torsion_4atoms.append(rot_bond_torsion_4atoms[k])
        torsion_4atoms.append(rot_bond_torsion_4atoms[k][::-1])

        if len(r_atoms) == 4 and -1 not in r_atoms:
            ar_torsion_angles.append(GetDihedral(conf, r_atoms))
            torsion_angles.append(GetDihedral(conf, r_atoms))
            torsion_angles.append(GetDihedral(conf, r_atoms[::-1]))
        else:
            ar_torsion_angles.append(None)
            torsion_angles.append(None)
            torsion_angles.append(None)
    ar_edge_index = np.stack(list(rot_bond_frag.values()), -1)[:, ar_order]
    ar_attach_index = np.stack(list(rot_bond_attach_atom_idx.values()), -1)[:, ar_order]
    frag_edge_index = np.array([src_frag_idx, dst_frag_idx], dtype=np.int64)
    frag_l_attach_index = np.array([src_l_atom_idx, dst_l_atom_idx], dtype=np.int64)
    return ar_edge_index, ar_attach_index, ar_torsion_4atoms, ar_torsion_angles, \
           frag_edge_index, frag_l_attach_index, torsion_4atoms, torsion_angles


def reconstruct_from_ar_index(all_frag_info, ar_edge_index, ar_attach_index, ar_torsion_4atoms, ar_torsion_angles,
                              return_all=False, von_mises=True):
    all_frags = [f['match_ref_frag'] for f in all_frag_info]
    all_query_frags = [deepcopy(f['aligned_query_frag']) for f in all_frag_info]
    # reconstruct for standardization
    recon_frag = all_frags[0]
    recon_frags = [recon_frag]
    # recon_frag_true = all_query_frags[0]
    acc_num_atoms = all_frags[0].GetNumAtoms()
    acc_global_idx = {0: np.arange(acc_num_atoms)}
    remap_dict = {}  # [src frag idx, src atom idx, dst frag idx, dst atom idx]
    new_bond_list = []

    recon_frag_true_list = []
    for fe_idx, (exist_frag_idx, new_frag_idx) in enumerate(ar_edge_index.T):
        src_atom_idx, dst_atom_idx = int(ar_attach_index[0][fe_idx]), int(ar_attach_index[1][fe_idx])
        cur_attach_idx = acc_global_idx[exist_frag_idx][src_atom_idx]
        new_attach_idx = dst_atom_idx
        # gather torsion atoms
        if -1 not in ar_torsion_4atoms[fe_idx]:
            cur_torsion_2atoms = []
            for i in ar_torsion_4atoms[fe_idx][:2]:
                if acc_global_idx[exist_frag_idx][i] != -1:
                    cur_torsion_2atoms.append(int(acc_global_idx[exist_frag_idx][i]))
                else:
                    remap_frag_idx, remap_atom_idx = remap_dict[f'{exist_frag_idx},{i}']
                    cur_torsion_2atoms.append(int(acc_global_idx[remap_frag_idx][remap_atom_idx]))

            new_torsion_2atoms = ar_torsion_4atoms[fe_idx][2:]
            if von_mises:
                recon_frag_new, torsion_4atoms = add_frag_to_mol(
                    recon_frag, all_frags[new_frag_idx], int(cur_attach_idx), int(new_attach_idx),
                    cur_torsion_2atoms, new_torsion_2atoms,
                    # ar_torsion_angles[fe_idx]
                )
                recon_frag_true = add_frag_to_mol(
                    recon_frag, all_query_frags[new_frag_idx], int(cur_attach_idx), int(new_attach_idx))
                recon_frag_true_list.append(recon_frag_true)
                # von mises matching
                torsion_angle = get_dihedral_vonMises(recon_frag_new,
                                                      recon_frag_new.GetConformer(), torsion_4atoms,
                                                      recon_frag_true.GetConformer().GetPositions())
                # print('new dihedral: ', torsion_angle)
                # print('ar torsion angles: ', ar_torsion_angles[fe_idx])

                # apply torsion angles
                SetDihedral(recon_frag_new.GetConformer(), torsion_4atoms, torsion_angle)
                recon_frag = recon_frag_new
            else:
                recon_frag_new, torsion_4atoms = add_frag_to_mol(
                    recon_frag, all_frags[new_frag_idx], int(cur_attach_idx), int(new_attach_idx),
                    cur_torsion_2atoms, new_torsion_2atoms,
                    ar_torsion_angles[fe_idx]
                )
                recon_frag = recon_frag_new

        else:
            raise ValueError
            # recon_frag = add_frag_to_mol(
            #     recon_frag, all_frags[new_frag_idx], int(cur_attach_idx), int(new_attach_idx)
            # )
        new_bond_list.append(recon_frag.GetBondWithIdx(recon_frag.GetNumBonds() - 1))
        recon_frags.append(recon_frag)
        # update acc global idx
        new_frag_num_atoms = all_frags[new_frag_idx].GetNumAtoms()
        acc_global_idx[new_frag_idx] = np.arange(acc_num_atoms - 1, acc_num_atoms + new_frag_num_atoms - 1)
        acc_global_idx[exist_frag_idx][src_atom_idx] = -1
        acc_global_idx[new_frag_idx][dst_atom_idx] = -1
        # renumber
        i = 0
        for k, v in acc_global_idx.items():
            new_v = []
            for idx in v:
                if idx != -1:
                    new_v.append(i)
                    i += 1
                else:
                    new_v.append(-1)
            acc_global_idx[k] = np.array(new_v)
        acc_num_atoms = recon_frag.GetNumAtoms()
        remap_dict[f'{exist_frag_idx},{src_atom_idx}'] = [new_frag_idx, get_neighbor(all_frags[new_frag_idx], dst_atom_idx)]
        remap_dict[f'{new_frag_idx},{dst_atom_idx}'] = [exist_frag_idx, get_neighbor(all_frags[exist_frag_idx], src_atom_idx)]

    # find cutting bond index
    bond_list = []
    bond_atom_index = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in recon_frag.GetBonds()]
    for src, dst in ar_edge_index.T:
        src_set = acc_global_idx[src]
        dst_set = acc_global_idx[dst]
        bond = [(a1, a2) for i, (a1, a2) in enumerate(bond_atom_index) if
                (a1 in src_set and a2 in dst_set) or (a1 in dst_set and a2 in src_set)]
        assert len(bond) == 1
        bond_list.append(bond[0])

    # find rotatable bonds
    # rot_bonds = []
    # for (i, j) in bond_list:
    #     atom_i = recon_frag.GetAtomWithIdx(i)
    #     n1_idx = [n.GetIdx() for n in atom_i.GetNeighbors() if n.GetIdx() != j and n.GetSymbol() != '*']
    #     atom_j = recon_frag.GetAtomWithIdx(j)
    #     n2_idx = [n.GetIdx() for n in atom_j.GetNeighbors() if n.GetIdx() != i and n.GetSymbol() != '*']
    #     rot_bonds.append([n1_idx[0], i, j, n2_idx[0]])
    # print('rot bonds: ', rot_bonds)
    # # align recon_mol and non-H recon_mol to get the new rotatable bonds in the same order
    # Chem.SanitizeMol(recon_frag)
    # no_dummy_recon_mol = replace_atom_in_mol(recon_frag, 0, 1)
    # no_dummy_recon_mol = Chem.RemoveAllHs(no_dummy_recon_mol)
    #
    # no_dummy_recon_true = replace_atom_in_mol(recon_frag_true, 0, 1)
    # no_dummy_recon_true = Chem.RemoveAllHs(no_dummy_recon_true)
    # Chem.SanitizeMol(no_dummy_recon_mol)
    # match = recon_frag.GetSubstructMatches(no_dummy_recon_mol)
    # assert len(match) == 1
    # match = match[0]
    # no_dummy_rot_bonds = [[match.index(idx) for idx in rb] for rb in rot_bonds]
    # print('no dummy rot bonds: ', no_dummy_rot_bonds)
    #
    # # align mol and non-H recon_mol to get the reference Z
    # Chem.SanitizeMol(mol)
    # match2 = mol.GetSubstructMatches(no_dummy_recon_mol)
    # assert len(match2) == 1
    # match2 = match2[0]
    # Z = mol.GetConformer().GetPositions()[match2, :]
    #
    # von mises matching
    # new_dihedrals = np.zeros(len(rot_bonds))
    # for idx, r in enumerate(rot_bonds):
    #     new_dihedrals[idx] = get_dihedral_vonMises(recon_frag,
    #                                                recon_frag.GetConformer(), r,
    #                                                recon_frag_true.GetConformer().GetPositions())
    # print('new dihedral: ', new_dihedrals)
    # print('ar torsion angles: ', ar_torsion_angles)
    # # apply torsion angles
    # for i in range(len(rot_bonds)):
    #     SetDihedral(recon_frag.GetConformer(), rot_bonds[i], new_dihedrals[i])

    # apply torsion angles by Von Mises matching
    # new_dihedrals = np.zeros(len(rotable_bonds))
    #     for idx, r in enumerate(rotable_bonds):
    #         new_dihedrals[idx] = get_dihedral_vonMises(mol_rdkit,
    #                                                    mol_rdkit.GetConformer(conf_id), r,
    #                                                    mol.GetConformer().GetPositions())
    #     mol_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, conf_id)
    if return_all:
        return recon_frag, recon_frags, bond_list, recon_frag_true_list
    else:
        return recon_frag, bond_list


# code from https://github.com/gcorso/torsional-diffusion/
def A_transpose_matrix(alpha):
    return np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]], dtype=np.double)


def S_vec(alpha):
    return np.array([[np.cos(alpha)], [np.sin(alpha)]], dtype=np.double)


def get_dihedral_vonMises(mol, conf, atom_idx, Z):
    Z = np.array(Z)
    v = np.zeros((2, 1))
    iAtom = mol.GetAtomWithIdx(atom_idx[1])
    jAtom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in iAtom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in jAtom.GetBonds():
            l = b2.GetOtherAtomIdx(j)
            if l == i:
                continue
            assert k != l
            s_star = S_vec(GetDihedralFromPointCloud(Z, (k, i, j, l)))
            a_mat = A_transpose_matrix(GetDihedral(conf, (k, i, j, k_0)) + GetDihedral(conf, (l_0, i, j, l)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.arctan2(v[1], v[0])


# def apply_changes(mol, values, rotable_bonds, conf_id):
#     opt_mol = copy.copy(mol)
#     [SetDihedral(opt_mol.GetConformer(conf_id), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]
#     return opt_mol

# def get_von_mises_rms(mol, mol_rdkit, rotable_bonds, conf_id):
#     new_dihedrals = np.zeros(len(rotable_bonds))
#     for idx, r in enumerate(rotable_bonds):
#         new_dihedrals[idx] = get_dihedral_vonMises(mol_rdkit,
#                                                    mol_rdkit.GetConformer(conf_id), r,
#                                                    mol.GetConformer().GetPositions())
#     mol_rdkit = apply_changes(mol_rdkit, new_dihedrals, rotable_bonds, conf_id)
#     return RMSD(mol_rdkit, mol, conf_id)


def get_frag_repr(mol, sg, vocab, von_mises=True):
    all_frag_info, frags, dummy_end, _ = extract_all_frag_info(mol, sg, vocab)
    if len(all_frag_info) > 1:
        ar_edge_index, ar_attach_index, ar_torsion_4atoms, ar_torsion_angles, _, _, _, _ = \
            build_frag_graph(mol, frags, all_frag_info, dummy_end)

        all_frags_idx = [f['frag_type'] for f in all_frag_info]
        recon_mol, recon_mols, cut_bond_list, recon_mols_true = reconstruct_from_ar_index(
            all_frag_info, ar_edge_index, ar_attach_index, ar_torsion_4atoms, ar_torsion_angles,
            von_mises=von_mises, return_all=True)
        # process one more time for recon mol
        recon_frag_info, recon_frags, dummy_end, _ = extract_all_frag_info(recon_mol, sg, vocab, cut_bond_list=cut_bond_list,
                                                         reorder=False, fix_frag_idx=all_frags_idx)
        _, _, _, _, frag_edge_index, l_attach_index, torsion_4atoms, torsion_angles = \
            build_frag_graph(recon_mol, recon_frags, recon_frag_info, dummy_end)
    else:
        # single fragment case
        recon_mol = all_frag_info[0]['match_ref_frag']
        recon_mols = [recon_mol]
        recon_frag_info = all_frag_info
        frag_edge_index, l_attach_index = np.empty([2, 0], dtype=np.int64), np.empty([2, 0], dtype=np.int64)
        torsion_4atoms = np.empty([0, 4], dtype=np.int64)
        torsion_angles = []

    Chem.Kekulize(recon_mol)
    # compute RMSD between recon_mol and original mol
    san_recon_mol = replace_atom_in_mol(recon_mol, 0, 1)
    san_recon_mol = Chem.RemoveAllHs(san_recon_mol)
    Chem.SanitizeMol(san_recon_mol)
    san_mol = deepcopy(mol)
    Chem.SanitizeMol(san_mol)
    rmsd = GetBestRMS(san_mol, san_recon_mol)  # directly compute RMSD

    # store necessary info
    ligand_data_dict = process_from_mol(recon_mol)
    frag_types = [f['frag_type'] for f in recon_frag_info]
    l_unique_attach_index = []
    for i, ((src_frag, dst_frag), (src_atom, dst_atom)) in enumerate(zip(frag_edge_index.T, l_attach_index.T)):
        u_att_index = [recon_frag_info[src_frag]['unique_dummy_mapping'][src_atom],
                       recon_frag_info[dst_frag]['unique_dummy_mapping'][dst_atom]]
        l_unique_attach_index.append(u_att_index)
    l_unique_attach_index = np.array(l_unique_attach_index).T
    g_torsion_4atoms = []
    for e_idx, (src, dst) in enumerate(frag_edge_index.T):
        g1 = [recon_frag_info[src]['g_all_atom_idx'][x] for x in torsion_4atoms[e_idx][:2]]
        g2 = [recon_frag_info[dst]['g_all_atom_idx'][x] for x in torsion_4atoms[e_idx][2:]]
        assert len(set(g1 + g2)) == 4
        g_torsion_4atoms.append(g1 + g2)

    # print('rmsd: ', rmsd)
    frag_dict = {
        'recon_mol': recon_mol,
        'frag_types': frag_types,
        'frag_edge_index': frag_edge_index,

        'l_attach_index': l_attach_index,
        'l_unique_attach_index': l_unique_attach_index,
        'all_unique_attach_index': [list(info['unique_dummy_dict'].keys()) for info in recon_frag_info],
        'l_torsion_4atoms': torsion_4atoms,
        'g_torsion_4atoms': g_torsion_4atoms,
        'torsion_angles': torsion_angles,
        'frag_rmsd': [x['rmsd'] for x in recon_frag_info],
        'rmsd': rmsd
    }
    # return frag_dict, all_frag_info, recon_frag_info, ligand_data_dict, recon_mols
    return frag_dict, recon_frag_info, ligand_data_dict
