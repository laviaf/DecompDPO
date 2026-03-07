import argparse
import copy
import json
import os
import re
import sys

import numpy as np
from rdkit import Chem, RDLogger
import torch
from tqdm.auto import tqdm

from utils import misc
from utils.evaluation import scoring_func
from utils.evaluation.docking_vina import VinaDockingTask


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def scoring(rdmol, protein_root, VinaDockingTask, scoring_func, ligand_file):
    vina_task = VinaDockingTask.from_generated_mol(
        copy.deepcopy(rdmol), ligand_file, protein_root=protein_root)
    dock_result = vina_task.run(mode='dock', exhaustiveness=16)
    score_only_results = vina_task.run(mode='score_only', exhaustiveness=16)
    minimize_results = vina_task.run(mode='minimize', exhaustiveness=16)
    vina_results = {
        'dock': dock_result,
        'score_only': score_only_results,
        'minimize': minimize_results
    }
    chem_results = scoring_func.get_chem(rdmol)
    info_dict = {
        'ligand_file': ligand_file,
        'chem_results': chem_results,
        'vina_results': vina_results
    }
    return info_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--item', type=int, default=0)
    parser.add_argument('--save_path', type=str, default=None)
    parser.add_argument('--protein_root', type=str, default='./data/test_set')
    parser.add_argument('--sample_res_path', type=str, default=None)
    args = parser.parse_args()

    if os.path.exists(args.save_path):
        sys.exit(0)

    sample_file = os.path.join(args.sample_res_path, 'result.pt')
    if not os.path.exists(sample_file):
        sys.exit(0)

    os.makedirs('tmp', exist_ok=True)
    sample_ligand = torch.load(sample_file)

    smiles_list = []
    ligand_idxs = []
    mols = []
    results = {}

    for ligand_idx in range(len(sample_ligand)):
        rdmol = sample_ligand[ligand_idx]['mol']
        smiles = sample_ligand[ligand_idx]['smiles']
        if rdmol is None or '.' in smiles:
            print('skipped:', ligand_idx)
            continue

        try:
            Chem.SanitizeMol(rdmol)
        except Chem.rdchem.AtomValenceException as e:
            err = e
            N4_valence = re.compile(
                u"Explicit valence for atom # ([0-9]{1,}) N, 4, is greater than permitted")
            index = N4_valence.findall(err.args[0])
            if len(index) > 0:
                rdmol.GetAtomWithIdx(int(index[0])).SetFormalCharge(1)
                Chem.SanitizeMol(rdmol)
        smiles_list.append(smiles)
        mols.append(rdmol)
        ligand_idxs.append(ligand_idx)

        try:
            res = scoring(rdmol, args.protein_root, VinaDockingTask,
                          scoring_func, sample_ligand[0]['ligand_filename'])
            results[ligand_idx] = res
        except:
            print("Scoring error for ligand_idx:", ligand_idx)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w') as f:
        f.write(json.dumps(results, cls=NpEncoder))
