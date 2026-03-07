import argparse
import os
import shutil
import sys
import time

import numpy as np
import torch
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from torch_geometric.data import Batch
from torch_scatter import scatter_mean

import utils.misc as misc
import utils.train as utils_train
import utils.transforms as trans
from utils.amber_energy import stat_amber_info
from datasets.pl_data import FOLLOW_BATCH
from datasets.pl_dpo_dataset_decompdpo_common import get_decomp_dataset
from models.decompdiff import DecompScorePosNet3D
from models.transitions import index_to_log_onehot

torch.multiprocessing.set_sharing_strategy('file_system')


def generate_arms_pair_mask(pair1_batch, pair2_batch, metric='qed', _lambda=0.1):
    """Generate arms-level preference mask for LocalDPO (decomposable objectives)."""
    all_frags_num = sum(pair1_batch.num_arms) + pair1_batch.num_arms.shape[0]
    pair1_mask = torch.zeros(all_frags_num).to(args.device)
    pair2_mask = torch.zeros(all_frags_num).to(args.device)
    frag_num = 0
    for data_id in range(len(pair1_batch.num_arms)):
        arms_list1 = pair1_batch.arms_list[data_id]
        arms_list2 = pair2_batch.arms_list[data_id]
        for i in range(len(arms_list1)):
            reward1, reward2 = arms_list1[i][metric], arms_list2[i][metric]
            amber1 = getattr(arms_list1[i], 'amber_energy', 0)
            amber2 = getattr(arms_list2[i], 'amber_energy', 0)
            if amber1 > 0 or amber2 > 0:
                print(f"arms_list1[i]: {arms_list1[i]}")
                print(f"arms_list2[i]: {arms_list2[i]}")
            if amber1 > 0:
                print(f"amber1: {amber1}")
            if amber2 > 0:
                print(f"amber2: {amber2}")
            if metric in ['score_only', 'vina_min']:
                reward1 *= -1
                reward2 *= -1
            reward1 -= _lambda * amber1
            reward2 -= _lambda * amber2
            if reward1 > reward2:
                pair1_mask[frag_num] = 1
                pair2_mask[frag_num] = -1
            elif reward1 < reward2:
                pair1_mask[frag_num] = -1
                pair2_mask[frag_num] = 1
            else:
                pair1_mask[frag_num] = 0
                pair2_mask[frag_num] = 0
            frag_num += 1

        if pair1_batch.num_scaffold[data_id] > 0:
            scaffold_list1 = -1 * pair1_batch.scaffold_list[data_id]
            scaffold_list2 = -1 * pair2_batch.scaffold_list[data_id]
            if len(scaffold_list2) == 0:
                pair1_mask[frag_num] = 1
                pair2_mask[frag_num] = -1
            elif len(scaffold_list1) == 0:
                pair1_mask[frag_num] = 0
                pair2_mask[frag_num] = 0
            else:
                reward1, reward2 = scaffold_list1[0][metric], scaffold_list2[0][metric]
                amber1 = getattr(scaffold_list1[0], 'amber_energy', 0)
                amber2 = getattr(scaffold_list2[0], 'amber_energy', 0)
                if metric in ['score_only', 'vina_min']:
                    reward1 *= -1
                    reward2 *= -1
                reward1 -= _lambda * amber1
                reward2 -= _lambda * amber2
                if reward1 > reward2:
                    pair1_mask[frag_num] = 1
                    pair2_mask[frag_num] = -1
                elif reward1 < reward2:
                    pair1_mask[frag_num] = -1
                    pair2_mask[frag_num] = 1
                else:
                    pair1_mask[frag_num] = 0
                    pair2_mask[frag_num] = 0
                frag_num += 1
        else:
            pair1_mask[frag_num] = 0
            pair2_mask[frag_num] = 0
            frag_num += 1

    return pair1_mask, pair2_mask


def generate_pair_mask(pair1_batch, pair2_batch, metric='qed', _lambda=0.1):
    """Generate molecule-level preference mask for GlobalDPO (non-decomposable objectives)."""
    reward1 = getattr(pair1_batch, metric, 0)
    reward2 = getattr(pair2_batch, metric, 0)
    amber1 = getattr(pair1_batch, 'amber_energy', 0)
    amber2 = getattr(pair2_batch, 'amber_energy', 0)
    if metric in ['score_only', 'vina_min']:
        reward1 *= -1
        reward2 *= -1

    # Amber Energy Constraint
    reward1 -= _lambda * amber1
    reward2 -= _lambda * amber2

    pair1_mask = ((reward1 > reward2) * 2 - 1).to(args.device)
    pair2_mask = ((reward1 <= reward2) * 2 - 1).to(args.device)
    reward_mask = reward1 == reward2
    pair1_mask[reward_mask] = 0
    pair2_mask[reward_mask] = 0

    return pair1_mask, pair2_mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/decompdpo/train.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--logdir', type=str, default='./logs_diffusion_full')
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--train_report_iter', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--reference_model_ckpt', type=str, default=None)
    parser.add_argument('--schedule_type', type=str, default=None,
                        choices=[None, 'sigmoid', 'linear', 'cosine'],
                        help='Linear beta schedule type (Section 3.4 in the paper)')
    args = parser.parse_args()

    # Load configs
    config = misc.load_config(args.config)
    config_name = os.path.basename(args.config)[:os.path.basename(args.config).rfind('.')]
    misc.seed_all(config.train.seed)
    fn = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

    # Override reference model checkpoint if provided
    if args.reference_model_ckpt is not None:
        config.dpo.reference_model_ckpt = args.reference_model_ckpt

    # Override batch settings if provided
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
        config.train.num_workers = args.batch_size
        config.train.n_acc_batch = max(1, int(4 / args.batch_size))

    # Logging
    log_dir = os.path.join(args.logdir, '%s_%s' % (config_name, fn))
    ckpt_dir = os.path.join(log_dir, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    vis_dir = os.path.join(log_dir, 'vis')
    os.makedirs(vis_dir, exist_ok=True)
    logger = misc.get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    logger.info(args)
    logger.info(config)
    shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    shutil.copytree('./models', os.path.join(log_dir, 'models'), dirs_exist_ok=True)

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_featurizer = trans.FeaturizeLigandAtom(
        config.data.transform.ligand_atom_mode,
        config.model.prior_types,
        has_recon_failed_data=config.data.transform.has_recon_failed_data
    )
    decomp_indicator = trans.AddDecompIndicator(
        max_num_arms=config.data.transform.max_num_arms,
        global_prior_index=ligand_featurizer.ligand_feature_dim,
        add_ord_feat=getattr(config.data.transform, 'add_ord_feat', True),
    )
    transform_list = [
        trans.ComputeLigandAtomNoiseDist(version=config.data.get('prior_mode', 'subpocket')),
        protein_featurizer,
        ligand_featurizer,
        decomp_indicator
    ]
    if config.data.transform.random_rot:
        transform_list.append(trans.RandomRotation())
    if getattr(config.model, 'bond_diffusion', False):
        transform_list.append(
            trans.FeaturizeLigandBond(mode=config.data.transform.ligand_bond_mode, set_bond_type=True)
        )
    if config.data.transform.amber_constrain:
        if not os.path.exists(config.data.transform.bond_info_path) or \
           not os.path.exists(config.data.transform.angle_info_path):
            stat_amber_info(config.data.transform.amber_dataset_config,
                            config.data.transform.bond_info_path,
                            config.data.transform.angle_info_path)
        transform_list.append(trans.AmberFeaturizer(
            config.data.transform.bond_info_path, config.data.transform.angle_info_path))
    transform = Compose(transform_list)

    # Datasets and loaders
    logger.info('Loading dataset...')
    dataset, subsets = get_decomp_dataset(
        config=config.data,
        transform=transform
    )
    train_set_high, train_set_low, test_set_high, test_set_low = \
        subsets['train']['high_set'], subsets['train']['low_set'], \
        subsets['test']['high_set'], subsets['test']['low_set']
    logger.info(f'Train set: {len(train_set_high)} Test set: {len(test_set_high)}')

    collate_exclude_keys = [
        'ligand_nbh_list', 'pocket_atom_masks', 'pocket_prior_masks',
        'scaffold_prior', 'arms_prior', 'ligand_atom_feature',
        'ligand_hybridization', 'ligand_file', 'arms_dict'
    ]
    dataloader_kwargs = dict(
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=config.train.num_workers,
        follow_batch=FOLLOW_BATCH,
        exclude_keys=collate_exclude_keys,
    )
    if config.train.num_workers > 0:
        dataloader_kwargs['prefetch_factor'] = 4
        dataloader_kwargs['persistent_workers'] = True
    train_high_loader = utils_train.inf_iterator(DataLoader(
        train_set_high, **dataloader_kwargs
    ))

    # Load reference model
    logger.info('Building reference model...')
    protein_feature_dim = sum([getattr(t, 'protein_feature_dim', 0) for t in transform_list])
    ligand_feature_dim = sum([getattr(t, 'ligand_feature_dim', 0) for t in transform_list])
    if getattr(config.model, 'add_valency_features', False):
        ligand_feature_dim += 3
    ref_model = DecompScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_feature_dim,
        ligand_atom_feature_dim=ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob,
        prior_bond_types=ligand_featurizer.bond_types_prob
    ).to(args.device)
    ckpt = torch.load(config.dpo.reference_model_ckpt, map_location=args.device)
    ref_model.load_state_dict(ckpt['model'])
    ref_model.eval()

    # Model
    logger.info('Building model...')
    model = DecompScorePosNet3D(
        config.model,
        protein_atom_feature_dim=protein_feature_dim,
        ligand_atom_feature_dim=ligand_feature_dim,
        num_classes=ligand_featurizer.ligand_feature_dim,
        prior_atom_types=ligand_featurizer.atom_types_prob,
        prior_bond_types=ligand_featurizer.bond_types_prob
    ).to(args.device)
    # Initialize from pretrained model
    model.load_state_dict(ckpt['model'])
    print(f'protein feature dim: {protein_feature_dim} '
          f'ligand feature dim: {ligand_feature_dim} ')
    logger.info(f'# trainable parameters: {misc.count_parameters(model) / 1e6:.4f} M')

    # Optimizer
    optimizer = utils_train.get_optimizer(config.train.optimizer, model)

    def get_batch_loss(config, batch, model, ref_model, protein_pos, prior_centers,
                       time_step=None, tight_loss=False):
        """Compute DPO loss for a batch (Section 3.2 in the paper)."""
        num_graphs = batch.protein_element_batch.max().item() + 1
        ligand_decomp_batch = batch.ligand_decomp_mask
        prior_stds = batch.ligand_decomp_stds
        batch_ligand = batch.ligand_element_batch
        ligand_pos = batch.ligand_pos
        ligand_v = batch.ligand_atom_feature_full
        ligand_fc_bond_type = getattr(batch, 'ligand_fc_bond_type', None)
        batch_ligand_bond = getattr(batch, 'ligand_fc_bond_type_batch', None)

        # 1. Sample noise levels
        if time_step is None:
            time_step, pt = model.sample_time(num_graphs, protein_pos.device, model.sample_time_method)
        a = model.alphas_cumprod.index_select(0, time_step)

        # 2. Perturb pos, v, (and bond)
        assert len(ligand_decomp_batch) == batch.ligand_decomp_num_atoms.sum().item()
        batch_noise_centers = prior_centers[ligand_decomp_batch]
        batch_noise_stds = prior_stds[ligand_decomp_batch]
        assert len(batch_noise_centers) == len(batch_ligand)
        assert len(batch_noise_stds) == len(batch_ligand)
        a_pos = a[batch_ligand].unsqueeze(-1)
        pos_noise = torch.zeros_like(ligand_pos)
        pos_noise.normal_()
        # Xt = a.sqrt() * X0 + (1-a).sqrt() * eps
        ligand_pos_perturbed = a_pos.sqrt() * (ligand_pos - batch_noise_centers) + \
                               (1.0 - a_pos).sqrt() * pos_noise * batch_noise_stds + batch_noise_centers
        # Vt = a * V0 + (1-a) / K
        log_ligand_v0 = index_to_log_onehot(ligand_v, model.num_classes)
        ligand_v_perturbed, log_ligand_vt = model.atom_type_trans.q_v_sample(
            log_ligand_v0, time_step, batch_ligand)

        if model.bond_diffusion:
            log_ligand_b0 = index_to_log_onehot(ligand_fc_bond_type, model.num_bond_classes)
            ligand_b_perturbed, log_ligand_bt = model.bond_type_trans.q_v_sample(
                log_ligand_b0, time_step, batch_ligand_bond)
        else:
            ligand_b_perturbed = None

        shared_kwargs = dict(
            protein_pos=protein_pos,
            protein_v=batch.protein_atom_feature.float(),
            batch_protein=batch.protein_element_batch,
            ligand_pos=batch.ligand_pos,
            ligand_v_aux=batch.ligand_atom_aux_feature.float(),
            batch_ligand=batch.ligand_element_batch,
            ligand_pos_perturbed=ligand_pos_perturbed,
            ligand_v_perturbed=ligand_v_perturbed,
            log_ligand_vt=log_ligand_vt,
            ligand_b_perturbed=ligand_b_perturbed,
            log_ligand_bt=log_ligand_bt,
            log_ligand_v0=log_ligand_v0,
            log_ligand_b0=log_ligand_b0,
            pos_noise=pos_noise,
            batch_noise_stds=batch_noise_stds,
            prior_centers=prior_centers,
            batch_prior=batch.ligand_decomp_centers_batch,
            ligand_fc_bond_index=getattr(batch, 'ligand_fc_bond_index', None),
            batch_ligand_bond=getattr(batch, 'ligand_fc_bond_type_batch', None),
            dpo=True,
            time_step=time_step
        )

        model_loss = model.get_diffusion_dpo_loss(**shared_kwargs)
        ref_loss = ref_model.get_diffusion_dpo_loss(**shared_kwargs)

        if tight_loss:
            # Direct KL between ref and DPO model
            log_ligand_v_recon_model = F.log_softmax(model_loss['pred_ligand_v'], dim=-1)
            log_v_model_prob = model.atom_type_trans.q_v_posterior(
                log_ligand_v_recon_model, log_ligand_vt, time_step, batch_ligand)
            log_ligand_v_recon_ref = F.log_softmax(ref_loss['pred_ligand_v'], dim=-1)
            log_v_ref_prob = model.atom_type_trans.q_v_posterior(
                log_ligand_v_recon_ref, log_ligand_vt, time_step, batch_ligand)
            loss_v = model.compute_v_Lt_atom_wise(
                log_v_model_prob=log_v_ref_prob, log_v0=log_ligand_v0,
                log_v_true_prob=log_v_model_prob, t=time_step, batch=batch_ligand)

            log_ligand_b_recon_model = model_loss['ligand_b_recon']
            log_b_model_prob = model.bond_type_trans.q_v_posterior(
                log_ligand_b_recon_model, log_ligand_bt, time_step, batch_ligand_bond)
            log_ligand_b_recon_ref = ref_loss['ligand_b_recon']
            log_b_ref_prob = model.bond_type_trans.q_v_posterior(
                log_ligand_b_recon_ref, log_ligand_bt, time_step, batch_ligand_bond)
            loss_bond = model.compute_v_Lt_atom_wise(
                log_v_model_prob=log_b_ref_prob, log_v0=log_ligand_b0,
                log_v_true_prob=log_b_model_prob, t=time_step, batch=batch_ligand_bond)

            loss_pos = (((model_loss['pred_ligand_pos'] - ref_loss['pred_ligand_pos']) ** 2) /
                        (batch_noise_stds ** 2)).sum(-1)

            return {
                'losses': {
                    'pos': loss_pos,
                    'v': loss_v,
                    'bond': loss_bond
                }
            }
        else:
            return model_loss, ref_loss

    def train(it):
        model.train()
        optimizer.zero_grad()
        try:
            for _ in range(config.train.n_acc_batch):
                high_batch = train_high_loader.__next__().to(args.device)
                time_step, pt = model.sample_time(
                    high_batch.protein_element_batch.max().item() + 1,
                    high_batch.protein_pos.device, model.sample_time_method)
                protein_noise = torch.randn_like(high_batch.protein_pos) * config.train.pos_noise_std
                gt_protein_pos = high_batch.protein_pos + protein_noise
                prior_centers_noise = torch.randn_like(high_batch.ligand_decomp_centers) * config.train.prior_noise_std
                prior_centers = high_batch.ligand_decomp_centers + prior_centers_noise
                model_high_loss, ref_high_loss = get_batch_loss(
                    config, high_batch, model, ref_model, gt_protein_pos, prior_centers,
                    time_step, tight_loss=False)

                low_batch = Batch.from_data_list(
                    [train_set_low[i] for i in high_batch.id],
                    follow_batch=FOLLOW_BATCH, exclude_keys=collate_exclude_keys
                ).to(args.device)
                protein_noise = torch.randn_like(low_batch.protein_pos) * config.train.pos_noise_std
                gt_protein_pos = low_batch.protein_pos + protein_noise
                prior_centers_noise = torch.randn_like(low_batch.ligand_decomp_centers) * config.train.prior_noise_std
                prior_centers = low_batch.ligand_decomp_centers + prior_centers_noise
                model_low_loss, ref_low_loss = get_batch_loss(
                    config, low_batch, model, ref_model, gt_protein_pos, prior_centers,
                    time_step, tight_loss=False)

                # Generate arms-level preference masks (LocalDPO for Vina)
                pair1_arm_vina_mask, pair2_arm_vina_mask = generate_arms_pair_mask(
                    high_batch, low_batch, 'vina_min', config.train.amber_lambda)
                # Generate molecule-level preference masks (GlobalDPO for QED and SA)
                pair1_qed_mask, pair2_qed_mask = generate_pair_mask(
                    high_batch, low_batch, 'qed', config.train.amber_lambda)
                pair1_sa_mask, pair2_sa_mask = generate_pair_mask(
                    high_batch, low_batch, 'sa', config.train.amber_lambda)

                batch_ligand = high_batch.ligand_element_batch
                batch_ligand_bond = getattr(high_batch, 'ligand_fc_bond_type_batch', None)

                # Compute DecompDPO loss (Equation 9 in the paper)
                qed_loss_dict, sa_loss_dict, vina_loss_dict = {}, {}, {}
                for key in config.train.loss_weights.keys():
                    if key == 'bond':
                        # Bond-level DPO loss (GlobalDPO only for QED and SA; not decomposable for Vina)
                        dpo_diff_qed_kl = \
                            pair1_qed_mask[high_batch.ligand_fc_bond_type_batch].mul(
                                model_high_loss['losses'][key] - ref_high_loss['losses'][key]) + \
                            pair2_qed_mask[low_batch.ligand_fc_bond_type_batch].mul(
                                model_low_loss['losses'][key] - ref_low_loss['losses'][key])
                        dpo_diff_sa_kl = \
                            pair1_sa_mask[high_batch.ligand_fc_bond_type_batch].mul(
                                model_high_loss['losses'][key] - ref_high_loss['losses'][key]) + \
                            pair2_sa_mask[low_batch.ligand_fc_bond_type_batch].mul(
                                model_low_loss['losses'][key] - ref_low_loss['losses'][key])

                        qed_loss = -1 * 1000 * 2 * config.train.beta * dpo_diff_qed_kl
                        sa_loss = -1 * 1000 * 2 * config.train.beta * dpo_diff_sa_kl
                        assert batch_ligand_bond is not None
                        qed_loss = scatter_mean(qed_loss, batch_ligand_bond, dim=0)
                        sa_loss = scatter_mean(sa_loss, batch_ligand_bond, dim=0)
                    else:
                        dpo_diff_qed_kl = \
                            pair1_qed_mask[high_batch.ligand_element_batch].mul(
                                model_high_loss['losses'][key] - ref_high_loss['losses'][key]) + \
                            pair2_qed_mask[low_batch.ligand_element_batch].mul(
                                model_low_loss['losses'][key] - ref_low_loss['losses'][key])
                        dpo_diff_sa_kl = \
                            pair1_sa_mask[high_batch.ligand_element_batch].mul(
                                model_high_loss['losses'][key] - ref_high_loss['losses'][key]) + \
                            pair2_sa_mask[low_batch.ligand_element_batch].mul(
                                model_low_loss['losses'][key] - ref_low_loss['losses'][key])
                        dpo_diff_vina_kl = \
                            pair1_arm_vina_mask[high_batch.ligand_decomp_mask].mul(
                                model_high_loss['losses'][key] - ref_high_loss['losses'][key]) + \
                            pair2_arm_vina_mask[low_batch.ligand_decomp_mask].mul(
                                model_low_loss['losses'][key] - ref_low_loss['losses'][key])

                        qed_loss = -1 * 1000 * 3 * config.train.beta * dpo_diff_qed_kl
                        sa_loss = -1 * 1000 * 3 * config.train.beta * dpo_diff_sa_kl
                        vina_loss = -1 * 1000 * 3 * config.train.beta * dpo_diff_vina_kl
                        assert batch_ligand is not None
                        qed_loss = scatter_mean(qed_loss, batch_ligand, dim=0)
                        sa_loss = scatter_mean(sa_loss, batch_ligand, dim=0)
                        vina_loss = scatter_mean(vina_loss, batch_ligand, dim=0)

                    qed_loss_dict[key] = qed_loss
                    sa_loss_dict[key] = sa_loss
                    if key != 'bond':
                        vina_loss_dict[key] = vina_loss

                # Linear beta schedule (Section 3.4)
                if args.schedule_type == 'sigmoid':
                    time_weight = torch.exp((time_step - config.model.num_diffusion_timesteps / 2) / 100) / \
                        (1 + torch.exp((time_step - config.model.num_diffusion_timesteps / 2) / 100))
                elif args.schedule_type == 'linear':
                    time_weight = time_step / config.model.num_diffusion_timesteps
                elif args.schedule_type == 'cosine':
                    time_weight = (torch.cos(
                        ((config.model.num_diffusion_timesteps - time_step) /
                         config.model.num_diffusion_timesteps) * torch.pi) + 1) / 2

                final_loss = {}
                qed_loss = utils_train.sum_weighted_losses(qed_loss_dict, config.train.loss_weights)
                sa_loss = utils_train.sum_weighted_losses(sa_loss_dict, config.train.loss_weights)
                vina_loss = utils_train.sum_weighted_losses(vina_loss_dict, config.train.loss_weights)
                if args.schedule_type is not None:
                    qed_loss = time_weight.mul(qed_loss)
                    sa_loss = time_weight.mul(sa_loss)
                    vina_loss = time_weight.mul(vina_loss)

                # Multi-objective loss: equal weight for QED, SA, Vina
                loss = 1 / 3 * (torch.mean(-F.logsigmoid(qed_loss)) +
                                torch.mean(-F.logsigmoid(sa_loss)) +
                                torch.mean(-F.logsigmoid(vina_loss)))
                final_loss['overall'] = loss
                for k in qed_loss_dict.keys():
                    if k != 'bond':
                        final_loss[k] = 1 / 3 * (torch.mean(qed_loss_dict[k]) +
                                                  torch.mean(sa_loss_dict[k]) +
                                                  torch.mean(vina_loss_dict[k]))
                    else:
                        final_loss[k] = 1 / 2 * (torch.mean(qed_loss_dict[k]) +
                                                  torch.mean(sa_loss_dict[k]))

                loss = loss / config.train.n_acc_batch
                loss.backward()
            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)
            optimizer.step()
            if it % args.train_report_iter == 0:
                utils_train.log_losses(final_loss, it, 'train', args.train_report_iter,
                                       logger, writer, others={
                    'grad': orig_grad_norm,
                    'lr': optimizer.param_groups[0]['lr']
                })
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print('| WARNING: ran out of memory, skipping batch')
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad
                torch.cuda.empty_cache()
            else:
                raise e

        except StopIteration:
            print("Finish 1 Epoch. Exit Training.")
            ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
            torch.save({
                'config': config,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'iteration': it,
            }, ckpt_path)
            sys.exit(0)

    try:
        best_loss, best_iter = None, None
        for it in range(1, config.train.max_iters + 1):
            train(it)
            if it % config.train.save_freq == 0 or it == config.train.max_iters:
                ckpt_path = os.path.join(ckpt_dir, '%d.pt' % it)
                torch.save({
                    'config': config,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'iteration': it,
                }, ckpt_path)
    except KeyboardInterrupt:
        logger.info('Terminating...')
