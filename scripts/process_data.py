"""Process sampled training data into preference pairs for DecompDPO training.

This script reads the evaluation results of sampled molecules and constructs
preference pairs (high/low quality) for each protein target. The scoring function
is defined as: S = QED + SA + Vina_Min / (-12).
"""
import os
import argparse
import torch
from collections import defaultdict
from tqdm import tqdm

INF = 999999

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='outputs/training_data',
                        help='Directory containing sampled training data with evaluations')
    parser.add_argument('--qed_w', type=float, default=1, help='Weight for QED in scoring')
    parser.add_argument('--sa_w', type=float, default=1, help='Weight for SA in scoring')
    parser.add_argument('--vina_w', type=float, default=1, help='Weight for Vina Min in scoring')
    parser.add_argument('--save_dir', type=str, default='data/training_pairs.pt',
                        help='Output path for processed preference pairs')
    parser.add_argument('--has_recon_failed_data', type=eval, default=True,
                        help='Include reconstruction-failed molecules as negative examples')
    args = parser.parse_args()

    sample_list = os.listdir(args.data_dir)
    res_list = defaultdict(list)
    recon_fail = 0

    for s in tqdm(sample_list):
        try:
            res = torch.load(os.path.join(args.data_dir, s, 'eval/metrics.pt'))
        except:
            continue

        if len(res) < 2:
            continue

        top_score, tail_score = 0, INF
        for r in res:
            score = r['chem_results']['qed'] * args.qed_w + r['chem_results']['sa'] * args.sa_w \
                + r['vina']['minimize'][0]['affinity'] * args.vina_w / -12
            if score > top_score:
                top_score = score
                top = r

            if score < tail_score:
                tail_score = score
                tail = r

        res_list[int(s.split('_')[-1])].append({
            'high': top,
            'low': tail
        })

        if args.has_recon_failed_data:
            if len(res) < 7:
                recon_fail += 1
                res = torch.load(os.path.join(args.data_dir, s, 'result.pt'))
                res = [r for r in res if r['mol'] is None]
                res_list[int(s.split('_')[-1])].append({
                    'high': top,
                    'low': res[0]
                })

    print(f'total pairs num: {recon_fail + len(res_list)}, recon fail num: {recon_fail}')
    torch.save(res_list, args.save_dir)
