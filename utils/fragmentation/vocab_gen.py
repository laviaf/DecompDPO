from tqdm.auto import tqdm
from rdkit import Chem
from sklearn_extra.cluster import KMedoids
import numpy as np
from sklearn.metrics import silhouette_score
import math
import pickle
import os
from copy import deepcopy
from .utils import get_cano_mol, get_align_map, get_clean_mol, replace_atom_in_mol, \
    fill_hydrogen_for_matching, get_mol_match, get_best_alignment, set_rdmol_positions
from collections import defaultdict
from rdkit.ML.Cluster import Butina
from rdkit.Chem.rdchem import BondType
from rdkit.Chem.rdMolAlign import GetBestRMS


def get_all_fragments(sg, mols, max_num=100, save_frag_collection=None):
    # random.shuffle(mols)
    frag_collection = {}
    for mol in tqdm(mols, desc='Fragmentation'):
        frags, _, _ = sg.fragmentize(mol)
        frags = [get_clean_mol(f) for f in Chem.GetMolFrags(frags, asMols=True)]

        for f in frags:
            base_mol = replace_atom_in_mol(f, src_atom=0, dst_atom=1)
            Chem.RemoveStereochemistry(base_mol)
            base_mol = Chem.RemoveAllHs(base_mol)
            base_smi = Chem.MolToSmiles(base_mol)
            smi = Chem.MolToSmiles(f)

            if base_smi not in frag_collection:
                frag_collection[base_smi] = {smi: [f]}
            else:
                if smi not in frag_collection[base_smi]:
                    frag_collection[base_smi][smi] = [f]
                elif len(frag_collection[base_smi][smi]) < max_num:
                    frag_collection[base_smi][smi].append(f)
    if save_frag_collection:
        pickle.dump(frag_collection, open(save_frag_collection, 'wb'))
        print('Dump original frag collection to ', save_frag_collection)
    return frag_collection


def find_cano_attach_atoms(aligned_dict):
    """
    Find all atoms that are able to connect dummy atoms, summarize them and return (consider co-occurrence)
    Besides, we can also assume all hydrogen atoms are attachable
    :param aligned_dict: aligned mol dict
    :return: {index of attaching atom: maximum number of dummy atoms}
    """
    cano_attach_atoms = {}
    for smi, mol_list in aligned_dict.items():
        for mol in mol_list:
            attach_atoms = defaultdict(int)
            for bond in mol.GetBonds():
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                if mol.GetAtomWithIdx(start).GetSymbol() == '*':
                    attach_atoms[end] += 1
                elif mol.GetAtomWithIdx(end).GetSymbol() == '*':
                    attach_atoms[start] += 1
            for k, v in attach_atoms.items():
                if k not in cano_attach_atoms:
                    cano_attach_atoms[k] = v
                else:
                    cano_attach_atoms[k] = max(cano_attach_atoms[k], v)
    cano_attach_atoms = dict(sorted(cano_attach_atoms.items()))
    return list(cano_attach_atoms.keys())


def get_unique_dummy_ids(base_dummy_mol):
    # find symmetry in fragments --> determine unique attaching atom index
    dummy_atom_ids = []
    for atom in base_dummy_mol.GetAtoms():
        if atom.GetSymbol() == '*':
            dummy_atom_ids.append(atom.GetIdx())
    self_matches = base_dummy_mol.GetSubstructMatches(base_dummy_mol, uniquify=False)
    unique_dummy_ids = {}
    cur_dummy_id = 0
    used_dummy_ids = set()
    for d_atom_id in dummy_atom_ids:
        if d_atom_id in used_dummy_ids:
            continue
        possible_ids = set([m[d_atom_id] for m in self_matches])
        unique_dummy_ids[cur_dummy_id] = possible_ids
        used_dummy_ids = used_dummy_ids.union(possible_ids)
        cur_dummy_id += 1
    assert sum([len(v) for v in unique_dummy_ids.values()]) == len(dummy_atom_ids)
    return unique_dummy_ids


def get_3d_unique_dummy_ids(mol, rms_thres=0.2):
    dummy_ids = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            dummy_ids.append(atom.GetIdx())
    pos = mol.GetConformer().GetPositions()
    dist_mat = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    dummy_dist = np.sort(dist_mat[dummy_ids])

    dist_mat = np.linalg.norm(dummy_dist[:, None, :] - dummy_dist[None, :, :], axis=-1)
    dmat = []
    for i in range(len(dist_mat)):
        for j in range(i):
            dmat.append(dist_mat[i][j])
    clusters = Butina.ClusterData(dmat, len(dist_mat), rms_thres, isDistData=True, reordering=True)
    unique_dummy_ids = {}
    for i, c in enumerate(clusters):
        unique_dummy_ids[i] = [dummy_ids[x] for x in c]
    return unique_dummy_ids


def get_base_dummy_mol(base_mol, cano_attach_atoms):
    base_dummy_mol = deepcopy(base_mol)
    dummy_rwmol = Chem.RWMol(base_dummy_mol)
    for atom_id, num_dummy_atoms in cano_attach_atoms.items():
        for _ in range(num_dummy_atoms):
            dummy_atom = Chem.Atom('*')
            dummy_rwmol.AddAtom(dummy_atom)
            dummy_rwmol.AddBond(atom_id, dummy_rwmol.GetNumAtoms() - 1, BondType.SINGLE)
    dummy_mol = dummy_rwmol.GetMol()
    # Chem.SanitizeMol(dummy_mol)
    return dummy_mol


def process_group_fragments(base_smi, sub_mols_dict):
    """
    Process fragments which have the same base smiles, including:
    1. align the base part (w/o dummy atoms) for all fragments
    2. find atom index which allows to attach dummy atoms
    3. fill hydrogen atoms to make all fragments have the same #atoms, ready for subsequent clustering
    :param base_smi: base smiles
    :param sub_mols_dict: mols dict (with dummy atoms)
    :return: processed
    """
    # align_frag_collection = {}
    # 1. align fragments
    base_mol = Chem.MolFromSmiles(base_smi)
    Chem.RemoveStereochemistry(base_mol)
    cano_order = tuple(zip(*sorted([(j, i) for i, j in enumerate(Chem.CanonicalRankAtoms(base_mol))])))[1]
    base_mol = Chem.RenumberAtoms(base_mol, cano_order)

    aligned_dict = {}
    for smi, mol_list in sub_mols_dict.items():
        aligned_mol_list = []
        for mol in mol_list:
            new_mol = deepcopy(mol)
            Chem.RemoveStereochemistry(new_mol)
            cano_order, _ = get_mol_match(new_mol, base_mol, 'cano_rank')
            aligned_mol = Chem.RenumberAtoms(new_mol,
                                             list(cano_order) + list(range(len(cano_order), mol.GetNumAtoms())))
            aligned_mol_list.append(aligned_mol)
        aligned_dict[smi] = aligned_mol_list

    # 2. find attaching atoms
    cano_attach_atoms = find_cano_attach_atoms(aligned_dict)

    # 3. fill Hs and align for the subsequent clustering
    all_mol_list = []
    for smi, mol_list in aligned_dict.items():
        cano_mol = None
        for mol in mol_list:
            new_mol = fill_hydrogen_for_matching(mol, base_mol, cano_attach_atoms, align=False)
            if cano_mol is None:
                cano_mol = deepcopy(new_mol)
            best_rms, best_match = get_best_alignment(cano_mol, new_mol)
            new_mol = Chem.RenumberAtoms(new_mol, best_match)
            # zero CoM
            old_pos = new_mol.GetConformer().GetPositions()
            new_pos = old_pos - old_pos.mean(0)
            new_mol = set_rdmol_positions(new_mol, new_pos)
            all_mol_list.append(new_mol)

    # check consistency of num atoms
    all_num_atoms = []
    for mol in all_mol_list:
        all_num_atoms.append(mol.GetNumAtoms())
    assert len(set(all_num_atoms)) == 1, base_smi

    # find symmetry in fragments --> determine unique attaching atom index
    # base_dummy_mol = get_base_dummy_mol(base_mol, cano_attach_atoms)
    base_dummy_mol = deepcopy(all_mol_list[0])
    dummy_atom_ids = []
    for atom in base_dummy_mol.GetAtoms():
        if atom.GetSymbol() == '*':
            dummy_atom_ids.append(atom.GetIdx())
    base_dummy_mol.RemoveAllConformers()
    # unique_dummy_ids = get_unique_dummy_ids(base_dummy_mol)

    return FragmentInfo(
        base_mol=base_mol, base_dummy_mol=base_dummy_mol,
        rep_mols=all_mol_list, attach_atom_list=cano_attach_atoms, dummy_ids=dummy_atom_ids
    )


def gen_dist_map(mols):
    if len(mols) < 1:
        return []

    nums = len(mols)
    dist_map = []
    for i in range(nums - 1):
        dist_map.append([])
        for j in range(i + 1, nums):
            dist_map[i].append(GetBestRMS(mols[i], mols[j]))
            # dist_map[i].append(AlignMol(mols[i], mols[j]))

    nums = len(dist_map) + 1
    pred_dst = []
    for dd in dist_map:
        pred_dst += dd
    pred_dst = np.array(pred_dst)
    pred_adj = np.zeros((nums, nums))
    pred_adj[np.triu(np.ones((nums, nums)), 1) == 1] = pred_dst
    pred_adj = pred_adj.T + pred_adj
    return pred_adj


def get_cluster(pred_adj, n):
    model = KMedoids(n_clusters=n, metric='precomputed', method='pam', init='heuristic', max_iter=300,
                     random_state=None)
    model.fit(pred_adj)
    try:
        score = silhouette_score(pred_adj, model.labels_, metric='precomputed', sample_size=None, random_state=None, )
    except ValueError:
        score = 0.0

    return model.medoid_indices_.tolist(), model.inertia_, score


def get_center_num(n):
    a = max(math.floor(math.sqrt((n - 1) / 10)), 1)
    return a + 1


def get_conf_vocab(frags, rms_thres=0.5, only_rms_cluster=True):
    # clustering 3D fragments
    if len(frags) == 1:
        return frags, np.zeros([1, 1]), 1
    vocab = []
    dist_map = gen_dist_map(frags)
    # max_cluster_num = max(math.floor(math.sqrt((len(dist_map) - 1) / 1)), 1)
    dmat = []
    for i in range(len(dist_map)):
        for j in range(i):
            dmat.append(dist_map[i][j])
    if rms_thres > 0:
        rms_clusters = Butina.ClusterData(dmat, len(dist_map), rms_thres, isDistData=True, reordering=True)

    if only_rms_cluster:
        assert rms_thres > 0
        max_cluster_num = len(rms_clusters)
        for c in rms_clusters:
            vocab.append(frags[c[0]])
    else:
        if rms_thres > 0:
            max_cluster_num = min(10, len(rms_clusters))
        else:
            num_clusters = max(math.floor(math.sqrt(len(dist_map))), 1)
            max_cluster_num = min(10, num_clusters)

        best_score = -2
        best_idx = []
        if max_cluster_num == 1 or frags[0].GetNumAtoms() <= 2:
            best_idx = [dist_map.sum(1).argmin()]
        else:
            for n in range(2, max_cluster_num + 1):
                idx, _, score = get_cluster(dist_map, n)
                if score > best_score:
                    best_score = score
                    # best_inertia = inertia
                    best_idx = idx
        for i in best_idx:
            vocab.append(frags[i])
    return vocab, dist_map, max_cluster_num


class FragmentInfo(object):
    def __init__(self, base_mol, base_dummy_mol, rep_mols, attach_atom_list, dummy_ids):
        self.base_mol = base_mol
        self.base_dummy_mol = base_dummy_mol
        self.rep_mols = rep_mols

        self.attach_atom_list = attach_atom_list
        self.dummy_ids = dummy_ids
        self.unique_dummy_ids_list = None
        self.unique_attach_ids_list = None  # [list(v)[0] for v in unique_dummy_ids.values()]
        self.features = None


class FragmentVocab(object):
    def __init__(self, sg, vocab_3d, vocab_mask):
        self.sg = sg
        self.vocab_3d = vocab_3d
        self.vocab_mask = vocab_mask

        # align fragments with same base mol
        # for smi, v in vocab_3d.items():
        #     cano_mol = get_cano_mol(v.rep_mols[0])
        #     cano_aMap = get_align_map(v.rep_mols[0], cano_mol)
        #     aligned_rep_mols = []
        #     for rep_mol in v.rep_mols:
        #         rep_aMap = get_align_map(rep_mol, cano_mol)
        #         atom_map = [(i, cano_aMap[1][rep_aMap[0][i]]) for i in range(rep_mol.GetNumAtoms())]
        #         # rmsd = AlignMol(query_frag, ref_frag, atomMap=atom_map)
        #         new_order = [q_at for (q_at, r_at) in sorted(atom_map, key=lambda x: x[1])]
        #         aligned_rep_mol = Chem.RenumberAtoms(rep_mol, new_order)
        #         aligned_rep_mols.append(aligned_rep_mol)
        #     v.rep_mols = aligned_rep_mols

        # find 3D unique fragments
        for v_info in self.vocab_3d.values():
            unique_dummy_ids_list = []
            unique_attach_ids_list = []
            for rep_mol in v_info.rep_mols:
                udummy_ids = get_3d_unique_dummy_ids(rep_mol)
                unique_dummy_ids_list.append(udummy_ids)
                unique_attach_ids_list.append([x[0] for x in udummy_ids.values()])
            v_info.unique_dummy_ids_list = unique_dummy_ids_list
            v_info.unique_attach_ids_list = unique_attach_ids_list
        self._compute_idx_to_frag()

    def _compute_idx_to_frag(self):
        self.idx_to_vocab_3d_frag = []
        for v_info in self.vocab_3d.values():
            for i in range(len(v_info.rep_mols)):
                info_dict = {
                    'rep_mol': v_info.rep_mols[i],
                    'unique_dummy_ids': v_info.unique_dummy_ids_list[i],
                    'unique_attach_ids': v_info.unique_attach_ids_list[i],
                    'attach_atom_list': v_info.attach_atom_list,
                    'dummy_ids': v_info.dummy_ids,
                    'features': v_info.features,
                }
                self.idx_to_vocab_3d_frag.append(info_dict)

    def __len__(self):
        return len(self.idx_to_vocab_3d_frag)

    def get_frags(self, key) -> FragmentInfo:
        return deepcopy(self.vocab_3d[key])

    def get_frag_idx_list(self, key):
        return self.vocab_mask[key]

    def get_frag_with_idx(self, idx):
        return deepcopy(self.idx_to_vocab_3d_frag[idx])


# if __name__ == '__main__':
#     # generate vocabulary
#     m = [x.rdmol for x in pickle.load(open('geom-qm9/train_data_40k.pkl', 'rb'))]
#     # m = [data.ligand_rdmol]
#     mols = []
#     for mol in m:
#         try:
#             Chem.SanitizeMol(mol)
#             mol = Chem.RemoveAllHs(mol, sanitize=True)
#             mols.append(mol)
#         except:
#             pass
#
#     frag_collection = get_all_fragments(mols)
#     vocab_2d = {smi: idx for idx, smi in enumerate(list(frag_collection.keys()))}
#     print(len(vocab_2d))
#     cur = 0
#     vocab_mask = {}
#     vocab_3d = {}
#     for smi in tqdm(frag_collection):
#         vocab_conf = get_conf_vocab(frag_collection[smi])
#         print(smi, len(vocab_conf))
#         vocab_mask[smi] = list(range(cur, cur + len(vocab_conf)))
#         cur += len(vocab_conf)
#         vocab_3d[smi] = vocab_conf
#     print(cur)
# pickle.dump(vocab_2d,open('vocab/vocab_2d.pkl','wb'))
# pickle.dump(vocab_3d,open('vocab/vocab_3d.pkl','wb'))
# pickle.dump(vocab_mask,open('vocab/vocab_3d_mask.pkl','wb'))
