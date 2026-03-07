import os 
import rdkit
from rdkit import Chem
from rdkit.Geometry import Point3D
from rdkit.Chem import AllChem
import torch  
from tqdm.auto import tqdm
from glob import glob
from queue import PriorityQueue
import numpy as np



root = "eval_results/decompdiff_baseline_2023_04_09__02_39_13_origin"


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


def decompose_generated_ligand(r):
    mask = np.array(r['decomp_mask'])
    mol = r['mol']
    arms = []

    for arm_idx in range(max(r['decomp_mask']) + 1):
        atom_indices = np.where(mask == arm_idx)[0].tolist()
        
        arm = get_submol_from_mol(mol, atom_indices)
        if arm is None:
            print(f"[fail] to extract submol (arm).")
            return None
        smi = Chem.MolToSmiles(arm)
        if "." in smi:
            print(f"[fail] incompleted arm: {smi}")
            return None 
        arms.append(arm)

    if -1 in r['decomp_mask']:
        arm_idx = -1
        atom_indices = np.where(mask == arm_idx)[0].tolist()
        
        arm = get_submol_from_mol(mol, atom_indices)
        if arm is None:
            print(f"[fail] to extract submol (arm).")
            return None
        smi = Chem.MolToSmiles(arm)
        if "." in smi:
            print(f"[fail] incompleted arm: {smi}")
            return None 
        arms.append(arm)
        
    return arms



if __name__ == "__main__":
    # debug_cnt = 0

    for data_id in range(100):
        path = glob(os.path.join(root, f"eval_{data_id:03d}*"))
        if len(path) == 0:
            print(f"missed: {data_id:03d}")
            continue 
        path = path[0]
        res = torch.load(path)
        q = PriorityQueue()
        for idx, r in enumerate(res):
            mol = r['mol']
            qed = r['chem_results']['qed']
            sa = r['chem_results']['sa']
            score = r['vina']['score_only'][0]['affinity']
            q.put((score, -qed, -sa, idx))
        top_k = 8
        if q.qsize() < top_k:
            print("too few generated samples.")
            exit()
        p = PriorityQueue()
        for k in range(top_k):
            score, neg_qed, neg_sa, idx = q.get()
            qed = -neg_qed
            sa = -neg_sa
            p.put((-qed-sa, idx))
        _, idx = p.get()
        # print(idx)

        r = res[idx]
        mol = r['mol']
        qed = r['chem_results']['qed']
        sa = r['chem_results']['sa']
        score = r['vina']['score_only'][0]['affinity']
        # print(idx, qed, sa, score)
        # print(r.keys())
        # print(r['decomp_mask'])
        arms = []
        mask = np.array(r['decomp_mask'])

        complete_flag = True
        for arm_idx in range(max(r['decomp_mask']) + 1):
            atom_indices = np.where(mask == arm_idx)[0].tolist()
            # print("atom_indices=", [a+1 for a in atom_indices])
            arm = get_submol_from_mol(mol, atom_indices)
            if arm is None:
                print(f"=====> {idx}: fail to extract submol (arm)")
                complete_flag = False 
                break
            smi = Chem.MolToSmiles(arm)
            if "." in smi:
                print(f"=====> {idx}: incompleted arm: {smi}")
                complete_flag = False 
                break
            arms.append(arm)
        if not complete_flag:
            continue 

        sdf_path = f"data/decompdiff_generated_arms/{data_id:03d}.sdf"
        print("Success! sdf_path =", sdf_path)
        with Chem.SDWriter(sdf_path) as w:
            for arm in arms:
                w.write(arm)
        