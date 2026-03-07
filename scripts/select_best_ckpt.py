"""Select the best checkpoint based on normalized success rate.

Success criteria: QED >= 0.25, SA >= 0.59, Vina Dock <= -8.18.
The best checkpoint is selected by maximizing: success_rate * complete_rate.
"""
import os
import sys
import argparse
import json
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,
                        help='Path to directory containing checkpoint evaluation results')
    args = parser.parse_args()

    path = args.path
    ckpt_list = os.listdir(path)

    all_success_rate = []
    all_complete_rate = []
    iterate = []
    best_norm_success, best_success, best_complete = 0, 0, 0
    best_ckpt = None

    for ckpt in ckpt_list:
        samples = os.listdir(os.path.join(path, ckpt))

        dock, score_only, minimize = [], [], []
        qed, sa = [], []
        success_rate, complete_rate = [], []
        data_ids = []

        for s in samples:
            success = []
            try:
                with open(os.path.join(path, ckpt, s, 'eval/result.json'), 'r') as f:
                    eval_res = json.load(f)
            except:
                print(s)
                continue

            data_ids.append(int(s.split('-')[-1]))
            for res in eval_res.values():
                if res['chem_results']['qed'] >= 0.25 and \
                   res['chem_results']['sa'] >= 0.59 and \
                   res['vina_results']['dock'][0]['affinity'] <= -8.18:
                    success.append(1)
                dock.append(res['vina_results']['dock'][0]['affinity'])
                score_only.append(res['vina_results']['score_only'][0]['affinity'])
                minimize.append(res['vina_results']['minimize'][0]['affinity'])
                qed.append(res['chem_results']['qed'])
                sa.append(res['chem_results']['sa'])
            success_rate.append(np.sum(success) / len(eval_res))
            complete_rate.append(len(eval_res) / 20)

        if len(success_rate) == 0:
            continue

        update_flag = False
        if np.mean(success_rate) * np.mean(complete_rate) > best_norm_success:
            best_success = np.mean(success_rate)
            best_complete = np.mean(complete_rate)
            best_norm_success = np.mean(success_rate) * np.mean(complete_rate)
            best_ckpt = ckpt
            update_flag = True

        if update_flag:
            best_qed, best_sa = qed, sa
            best_score_only, best_minimize, best_dock = score_only, minimize, dock

        iter_id = int(ckpt.split('_')[-1])
        iterate.append(iter_id)
        all_success_rate.append(np.mean(success_rate))
        all_complete_rate.append(np.mean(complete_rate))

    if best_ckpt is None:
        print('No valid checkpoints found.')
        sys.exit(1)

    print(f'Best checkpoint: {best_ckpt}')
    print(f'Success rate: {best_success:.4f}')
    print(f'Complete rate: {best_complete:.4f}')
    print(f'QED    - Mean: {np.mean(best_qed):.4f}, Median: {np.median(best_qed):.4f}')
    print(f'SA     - Mean: {np.mean(best_sa):.4f}, Median: {np.median(best_sa):.4f}')
    print(f'Dock   - Mean: {np.mean(best_dock):.4f}, Median: {np.median(best_dock):.4f}')
    print(f'Score  - Mean: {np.mean(best_score_only):.4f}, Median: {np.median(best_score_only):.4f}')
    print(f'MinVina- Mean: {np.mean(best_minimize):.4f}, Median: {np.median(best_minimize):.4f}')
