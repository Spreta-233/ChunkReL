# -*-coding:utf8-*-

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import copy
import math
from itertools import combinations

from dataset import single_task_dataset
from functions import loss_functions


def random_select(id2cnt, select_size):
    selected_id2prob = {}
    all_ids = list(id2cnt.keys())
    selected_ids = set(random.sample(all_ids, select_size))
    for d_id in selected_ids:
        selected_id2prob[d_id] = 1.0
    return selected_ids, selected_id2prob


def add_new_data(data_file, new_data):
    ori_data = []
    ori_ids = set()
    if os.path.exists(data_file):
        with open(data_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    d_id = di[0]
                    ori_ids.add(int(d_id))
                    ori_data.append(di)
                except EOFError:
                    break
    all_data = ori_data
    for di in new_data:
        d_id = int(di[0])
        if d_id not in ori_ids:
            all_data.append(di)
    random.shuffle(all_data)
    with open(data_file, 'wb') as fw:
        for di in all_data:
            pickle.dump(di, fw)


def reduce_snapshot_losses(losses, snapshot_reduce='quantile', snapshot_quantile=0.2):
    losses = np.asarray(losses, dtype=np.float32)
    if losses.size == 0:
        raise ValueError('snapshot losses must not be empty')
    if snapshot_reduce == 'min':
        return float(np.min(losses))
    if snapshot_reduce == 'median':
        return float(np.median(losses))
    if snapshot_reduce == 'quantile':
        return float(np.quantile(losses, float(snapshot_quantile)))
    raise ValueError('Invalid snapshot_reduce: ' + str(snapshot_reduce))


def compute_snapshot_proxy_loss(losses, snapshot_reduce='quantile', snapshot_quantile=0.2,
                                agreement=None, agreement_lambda=0.0):
    base_loss = reduce_snapshot_losses(
        losses=losses,
        snapshot_reduce=snapshot_reduce,
        snapshot_quantile=snapshot_quantile
    )
    if float(agreement_lambda) > 0.0:
        if agreement is None:
            raise ValueError('agreement is required when snapshot agreement lambda > 0')
        base_loss += float(agreement_lambda) * (1.0 - float(agreement))
    return float(base_loss)


def _validate_snapshot_calibration_quantiles(alpha, beta):
    alpha = float(alpha)
    beta = float(beta)
    if alpha < 0.0 or alpha > 1.0 or beta < 0.0 or beta > 1.0:
        raise ValueError('snapshot calibration alpha and beta must be in [0, 1]')
    if beta <= alpha:
        raise ValueError('snapshot_calib_beta must be greater than snapshot alpha')
    return alpha, beta


def percentile_rank_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.asarray([], dtype=np.float32)
    if values.size == 1 or float(np.max(values) - np.min(values)) <= 1e-12:
        return np.full(values.shape, 0.5, dtype=np.float32)
    order = np.argsort(values, kind='mergesort')
    ranks = np.zeros(values.shape[0], dtype=np.float32)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and abs(float(values[order[end]]) - float(values[order[start]])) <= 1e-12:
            end += 1
        avg_rank = 0.5 * float(start + end - 1)
        for pos in range(start, end):
            ranks[order[pos]] = avg_rank / float(len(order) - 1)
        start = end
    return ranks


def compute_snapshot_confidence_risk(conf, conf_gate='linear', conf_gamma=1.0, conf_eps=1e-6,
                                     conf_log_lambda=0.2, lowconf_mask=False,
                                     lowvar_tail_score=0.0, lowvar_eta=0.02,
                                     not_highconf_mask=False):
    conf = float(conf)
    conf_gate = str(conf_gate)
    if conf_gate == 'linear':
        return min(max(1.0 - conf, 0.0), 1.0)
    if conf_gate == 'focal_log':
        conf_eps = float(conf_eps)
        if conf_eps <= 0.0:
            raise ValueError('snapshot_calib_conf_eps must be > 0')
        safe_conf = max(conf, conf_eps)
        risk = ((1.0 - conf) ** float(conf_gamma)) * (-math.log(safe_conf)) / math.log(2.0)
        return min(max(float(risk), 0.0), 1.0)
    if conf_gate == 'linear_log_residual':
        conf_eps = float(conf_eps)
        conf_log_lambda = float(conf_log_lambda)
        if conf_eps <= 0.0 or conf_eps >= 1.0:
            raise ValueError(
                'snapshot_calib_conf_eps must be in (0, 1) for linear_log_residual')
        if conf_log_lambda < 0.0:
            raise ValueError('snapshot_calib_conf_log_lambda must be >= 0')
        safe_conf = max(conf, conf_eps)
        log_residual = (-math.log(safe_conf)) / (-math.log(conf_eps))
        risk = (1.0 - conf) * (1.0 + conf_log_lambda * log_residual)
        return min(max(float(risk), 0.0), 1.0)
    if conf_gate == 'linear_log_residual_lowvar_tail':
        gate_llr = compute_snapshot_confidence_risk(
            conf=conf,
            conf_gate='linear_log_residual',
            conf_eps=conf_eps,
            conf_log_lambda=conf_log_lambda
        )
        lowvar_eta = float(lowvar_eta)
        if lowvar_eta < 0.0:
            raise ValueError('snapshot_calib_lowvar_eta must be >= 0')
        tail_score = min(max(float(lowvar_tail_score), 0.0), 1.0)
        residual = lowvar_eta * float(bool(lowconf_mask)) * tail_score * (1.0 - gate_llr)
        return min(max(float(gate_llr + residual), 0.0), 1.0)
    if conf_gate == 'linear_log_residual_protected_lowvar_tail':
        gate_llr = compute_snapshot_confidence_risk(
            conf=conf,
            conf_gate='linear_log_residual',
            conf_eps=conf_eps,
            conf_log_lambda=conf_log_lambda
        )
        lowvar_eta = float(lowvar_eta)
        if lowvar_eta < 0.0:
            raise ValueError('snapshot_calib_lowvar_eta must be >= 0')
        tail_score = min(max(float(lowvar_tail_score), 0.0), 1.0)
        residual = (
            lowvar_eta * float(bool(not_highconf_mask)) *
            tail_score * (1.0 - gate_llr)
        )
        return min(max(float(gate_llr + residual), 0.0), 1.0)
    raise ValueError('Invalid snapshot_calib_conf_gate: ' + str(conf_gate))


def compute_snapshot_lowvar_tail_context(conf_values, var_norm_values,
                                         lowconf_q=0.20, lowvar_tail_q=0.05):
    conf_values = np.asarray(conf_values, dtype=np.float64)
    var_norm_values = np.asarray(var_norm_values, dtype=np.float64)
    if conf_values.size != var_norm_values.size:
        raise ValueError('confidence and VarNorm values must have the same length')
    if conf_values.size == 0:
        return {
            'lowconf_threshold': 0.0,
            'lowconf_mask': np.asarray([], dtype=np.bool_),
            'var_rank_pct': np.asarray([], dtype=np.float64),
            'lowvar_tail_score': np.asarray([], dtype=np.float64)
        }

    lowconf_q = float(lowconf_q)
    lowvar_tail_q = float(lowvar_tail_q)
    if lowconf_q < 0.0 or lowconf_q > 1.0:
        raise ValueError('snapshot_calib_lowconf_q must be in [0, 1]')
    if lowvar_tail_q > 1.0:
        raise ValueError('snapshot_calib_lowvar_tail_q must be <= 1')

    lowconf_threshold = float(np.quantile(conf_values, lowconf_q))
    lowconf_mask = conf_values <= lowconf_threshold
    sample_count = int(var_norm_values.size)
    if sample_count <= 1 or float(np.max(var_norm_values) - np.min(var_norm_values)) <= 1e-12:
        var_rank_pct = np.ones(var_norm_values.shape, dtype=np.float64)
    else:
        order = np.argsort(var_norm_values, kind='mergesort')
        ranks = np.empty(sample_count, dtype=np.float64)
        ranks[order] = np.arange(sample_count, dtype=np.float64)
        var_rank_pct = ranks / float(max(sample_count - 1, 1))

    if lowvar_tail_q <= 0.0:
        lowvar_tail_score = np.zeros(var_norm_values.shape, dtype=np.float64)
    else:
        lowvar_tail_score = np.clip(
            (lowvar_tail_q - var_rank_pct) / lowvar_tail_q,
            0.0,
            1.0
        )
    return {
        'lowconf_threshold': lowconf_threshold,
        'lowconf_mask': lowconf_mask,
        'var_rank_pct': var_rank_pct,
        'lowvar_tail_score': lowvar_tail_score
    }


def compute_snapshot_protected_lowvar_tail_context(
        conf_values, var_norm_values, highconf_protect_q=0.80,
        lowvar_tail_q=0.05):
    conf_values = np.asarray(conf_values, dtype=np.float64)
    var_norm_values = np.asarray(var_norm_values, dtype=np.float64)
    if conf_values.size != var_norm_values.size:
        raise ValueError('confidence and VarNorm values must have the same length')
    if conf_values.size == 0:
        return {
            'highconf_threshold': 0.0,
            'not_highconf_mask': np.asarray([], dtype=np.bool_),
            'var_rank_pct': np.asarray([], dtype=np.float64),
            'lowvar_tail_score': np.asarray([], dtype=np.float64)
        }

    highconf_protect_q = float(highconf_protect_q)
    lowvar_tail_q = float(lowvar_tail_q)
    if highconf_protect_q < 0.0 or highconf_protect_q > 1.0:
        raise ValueError('snapshot_calib_highconf_protect_q must be in [0, 1]')
    if lowvar_tail_q > 1.0:
        raise ValueError('snapshot_calib_lowvar_tail_q must be <= 1')

    highconf_threshold = float(np.quantile(conf_values, highconf_protect_q))
    not_highconf_mask = conf_values < highconf_threshold
    sample_count = int(var_norm_values.size)
    if sample_count <= 1 or float(np.max(var_norm_values) - np.min(var_norm_values)) <= 1e-12:
        var_rank_pct = np.ones(var_norm_values.shape, dtype=np.float64)
    else:
        order = np.argsort(var_norm_values, kind='mergesort')
        ranks = np.empty(sample_count, dtype=np.float64)
        ranks[order] = np.arange(sample_count, dtype=np.float64)
        var_rank_pct = ranks / float(max(sample_count - 1, 1))

    if lowvar_tail_q <= 0.0:
        lowvar_tail_score = np.zeros(var_norm_values.shape, dtype=np.float64)
    else:
        lowvar_tail_score = np.clip(
            (lowvar_tail_q - var_rank_pct) / lowvar_tail_q,
            0.0,
            1.0
        )
    return {
        'highconf_threshold': highconf_threshold,
        'not_highconf_mask': not_highconf_mask,
        'var_rank_pct': var_rank_pct,
        'lowvar_tail_score': lowvar_tail_score
    }


def compute_snapshot_dynamic_calibration_risk(conf_risk, var_risk, forg_risk, tau=0.5,
                                              use_confidence=False, use_variability=False,
                                              use_forgetting=False):
    use_confidence = bool(use_confidence)
    use_variability = bool(use_variability)
    use_forgetting = bool(use_forgetting)
    ablation_mode = use_confidence or use_variability or use_forgetting
    if not ablation_mode:
        risk = float(conf_risk) * float(var_risk) + float(tau) * (float(conf_risk) * float(forg_risk))
    elif use_confidence and not use_variability and not use_forgetting:
        risk = float(conf_risk)
    elif use_variability and not use_confidence and not use_forgetting:
        risk = float(var_risk)
    elif use_forgetting and not use_confidence and not use_variability:
        risk = float(forg_risk)
    elif use_confidence and use_variability and not use_forgetting:
        risk = float(conf_risk) * float(var_risk)
    elif use_confidence and use_forgetting and not use_variability:
        risk = float(conf_risk) * float(forg_risk)
    elif use_variability and use_forgetting and not use_confidence:
        risk = float(var_risk) * float(forg_risk)
    else:
        risk = float(conf_risk) * float(var_risk) + float(tau) * (float(conf_risk) * float(forg_risk))
    return min(max(float(risk), 0.0), 1.0)


def compute_dynamic_calibrated_snapshot_ref_loss(snapshot_losses, p_trues, corrects,
                                                 alpha=0.1, beta=0.2, rho_max=0.1,
                                                 tau=0.5, var_norm=None,
                                                 use_confidence=False, use_variability=False,
                                                 use_forgetting=False,
                                                 conf_gate='linear', conf_gamma=1.0, conf_eps=1e-6,
                                                 return_details=False, conf_log_lambda=0.2,
                                                 lowconf_mask=False, lowvar_tail_score=0.0,
                                                 lowvar_eta=0.02,
                                                 not_highconf_mask=False):
    alpha, beta = _validate_snapshot_calibration_quantiles(alpha, beta)
    losses = np.asarray(snapshot_losses, dtype=np.float32)
    p_trues = np.asarray(p_trues, dtype=np.float32)
    corrects = np.asarray(corrects, dtype=np.float32)
    if losses.size == 0:
        raise ValueError('snapshot losses must not be empty')
    if losses.size != p_trues.size or losses.size != corrects.size:
        raise ValueError('snapshot losses, p_trues, and corrects must have the same length')
    if float(rho_max) < 0.0:
        raise ValueError('snapshot_calib_rho_max must be >= 0')
    q_alpha = float(np.quantile(losses, alpha))
    q_beta = float(np.quantile(losses, beta))
    conf = float(np.mean(p_trues))
    var = float(np.std(p_trues))
    var_norm = 0.5 if var_norm is None else float(var_norm)
    var_norm = min(max(var_norm, 0.0), 1.0)
    forget_count = 0
    for k in range(1, corrects.size):
        if int(corrects[k - 1]) == 1 and int(corrects[k]) == 0:
            forget_count += 1
    forg = float(forget_count) / float(max(corrects.size - 1, 1))
    ablation_mode = bool(use_confidence) or bool(use_variability) or bool(use_forgetting)
    if ablation_mode:
        gate_llr = compute_snapshot_confidence_risk(
            conf=conf,
            conf_gate='linear_log_residual',
            conf_eps=conf_eps,
            conf_log_lambda=conf_log_lambda
        ) if conf_gate in (
            'linear_log_residual_lowvar_tail',
            'linear_log_residual_protected_lowvar_tail'
        ) else 0.0
        conf_risk = compute_snapshot_confidence_risk(
            conf=conf,
            conf_gate=conf_gate,
            conf_gamma=conf_gamma,
            conf_eps=conf_eps,
            conf_log_lambda=conf_log_lambda,
            lowconf_mask=lowconf_mask,
            lowvar_tail_score=lowvar_tail_score,
            lowvar_eta=lowvar_eta,
            not_highconf_mask=not_highconf_mask
        )
    else:
        conf_risk = 1.0 - conf
        gate_llr = conf_risk
    if conf_gate not in (
            'linear_log_residual_lowvar_tail',
            'linear_log_residual_protected_lowvar_tail'):
        gate_llr = conf_risk
    lowvar_tail_residual = conf_risk - gate_llr
    protected_lowvar_residual = (
        lowvar_tail_residual
        if conf_gate == 'linear_log_residual_protected_lowvar_tail'
        else 0.0
    )
    var_risk = 1.0 - var_norm
    forg_risk = forg
    risk = compute_snapshot_dynamic_calibration_risk(
        conf_risk=conf_risk,
        var_risk=var_risk,
        forg_risk=forg_risk,
        tau=tau,
        use_confidence=use_confidence,
        use_variability=use_variability,
        use_forgetting=use_forgetting
    )
    rho = float(rho_max) * risk
    ref_loss = q_alpha + rho * (q_beta - q_alpha)
    details = {
        'q_alpha': q_alpha,
        'q_beta': q_beta,
        'conf': conf,
        'var': var,
        'var_norm': var_norm,
        'forg': forg,
        'conf_risk': conf_risk,
        'var_risk': var_risk,
        'forg_risk': forg_risk,
        'ablation_mode': ablation_mode,
        'use_confidence': bool(use_confidence),
        'use_variability': bool(use_variability),
        'use_forgetting': bool(use_forgetting),
        'conf_gate': str(conf_gate),
        'conf_gamma': float(conf_gamma),
        'conf_eps': float(conf_eps),
        'conf_log_lambda': float(conf_log_lambda),
        'lowconf_mask': bool(lowconf_mask),
        'not_highconf_mask': bool(not_highconf_mask),
        'lowvar_tail_score': float(lowvar_tail_score),
        'lowvar_eta': float(lowvar_eta),
        'lowvar_tail_residual': float(lowvar_tail_residual),
        'protected_lowvar_residual': float(protected_lowvar_residual),
        'gate_llr': float(gate_llr),
        'gate_final': float(conf_risk),
        'risk': risk,
        'rho': rho,
        'ref_loss': float(ref_loss)
    }
    if bool(return_details):
        return float(ref_loss), details
    return float(ref_loss)


def compute_dynamic_calibrated_snapshot_ref_loss_dic(id2losses, id2p_trues, id2corrects,
                                                     alpha=0.1, beta=0.2, rho_max=0.1,
                                                     tau=0.5, norm_scope='task',
                                                     use_confidence=False, use_variability=False,
                                                     use_forgetting=False,
                                                     conf_gate='linear', conf_gamma=1.0, conf_eps=1e-6,
                                                     conf_log_lambda=0.2, lowconf_q=0.20,
                                                     lowvar_tail_q=0.05, lowvar_eta=0.02,
                                                     highconf_protect_q=0.80):
    alpha, beta = _validate_snapshot_calibration_quantiles(alpha, beta)
    if norm_scope != 'task':
        raise ValueError('unsupported snapshot_calib_norm_scope: ' + str(norm_scope))
    d_ids = sorted(id2losses.keys())
    vars_by_id = {}
    for d_id in d_ids:
        vars_by_id[d_id] = float(np.std(np.asarray(id2p_trues[d_id], dtype=np.float32)))
    var_norm_values = percentile_rank_normalize([vars_by_id[d_id] for d_id in d_ids])
    id2var_norm = {
        d_id: float(var_norm_values[idx])
        for idx, d_id in enumerate(d_ids)
    }
    ablation_mode = bool(use_confidence) or bool(use_variability) or bool(use_forgetting)
    tail_gate_active = (
        ablation_mode and bool(use_confidence) and
        str(conf_gate) == 'linear_log_residual_lowvar_tail'
    )
    protected_tail_gate_active = (
        ablation_mode and bool(use_confidence) and
        str(conf_gate) == 'linear_log_residual_protected_lowvar_tail'
    )
    tail_context = {
        'lowconf_mask': np.zeros(len(d_ids), dtype=np.bool_),
        'lowvar_tail_score': np.zeros(len(d_ids), dtype=np.float64)
    }
    if tail_gate_active:
        tail_context = compute_snapshot_lowvar_tail_context(
            conf_values=[
                float(np.mean(np.asarray(id2p_trues[d_id], dtype=np.float32)))
                for d_id in d_ids
            ],
            var_norm_values=[id2var_norm[d_id] for d_id in d_ids],
            lowconf_q=lowconf_q,
            lowvar_tail_q=lowvar_tail_q
        )
    protected_tail_context = {
        'not_highconf_mask': np.zeros(len(d_ids), dtype=np.bool_),
        'lowvar_tail_score': np.zeros(len(d_ids), dtype=np.float64)
    }
    if protected_tail_gate_active:
        protected_tail_context = compute_snapshot_protected_lowvar_tail_context(
            conf_values=[
                float(np.mean(np.asarray(id2p_trues[d_id], dtype=np.float32)))
                for d_id in d_ids
            ],
            var_norm_values=[id2var_norm[d_id] for d_id in d_ids],
            highconf_protect_q=highconf_protect_q,
            lowvar_tail_q=lowvar_tail_q
        )
    lowconf_mask_values = np.asarray(tail_context['lowconf_mask'], dtype=np.bool_)
    not_highconf_mask_values = np.asarray(
        protected_tail_context['not_highconf_mask'], dtype=np.bool_)
    if protected_tail_gate_active:
        lowvar_tail_scores = np.asarray(
            protected_tail_context['lowvar_tail_score'], dtype=np.float64)
    else:
        lowvar_tail_scores = np.asarray(
            tail_context['lowvar_tail_score'], dtype=np.float64)
    lowvar_tail_mask = lowvar_tail_scores > 0.0
    ref_loss_dic = {}
    diagnostics = {
        'alpha': alpha,
        'beta': beta,
        'rho_max': float(rho_max),
        'tau': float(tau),
        'norm_scope': norm_scope,
        'ablation_mode': ablation_mode,
        'use_confidence': bool(use_confidence),
        'use_variability': bool(use_variability),
        'use_forgetting': bool(use_forgetting),
        'conf_gate': str(conf_gate),
        'conf_gamma': float(conf_gamma),
        'conf_eps': float(conf_eps),
        'conf_log_lambda': float(conf_log_lambda),
        'lowconf_q': float(lowconf_q),
        'lowvar_tail_q': float(lowvar_tail_q),
        'lowvar_eta': float(lowvar_eta),
        'highconf_protect_q': float(highconf_protect_q),
        'lowconf_count': int(np.sum(lowconf_mask_values)),
        'lowvar_tail_count': int(np.sum(lowvar_tail_mask)),
        'joint_lowconf_lowvar_count': int(np.sum(
            np.logical_and(lowconf_mask_values, lowvar_tail_mask))),
        'highconf_protected_count': int(np.sum(
            np.logical_not(not_highconf_mask_values)
        )) if protected_tail_gate_active else 0,
        'not_highconf_count': int(np.sum(
            not_highconf_mask_values
        )) if protected_tail_gate_active else 0,
        'active_protected_lowvar_count': int(np.sum(
            np.logical_and(not_highconf_mask_values, lowvar_tail_mask)
        )) if protected_tail_gate_active else 0,
        'conf': [],
        'var': [],
        'var_norm': [],
        'forg': [],
        'conf_risk': [],
        'var_risk': [],
        'forg_risk': [],
        'risk': [],
        'final risk': [],
        'rho': [],
        'q_alpha': [],
        'q_beta': [],
        'ref_loss': [],
        'lowvar_tail_residual': [],
        'protected_lowvar_residual': [],
        'gate_llr': [],
        'gate_final': []
    }
    for idx, d_id in enumerate(d_ids):
        ref_loss, details = compute_dynamic_calibrated_snapshot_ref_loss(
            snapshot_losses=id2losses[d_id],
            p_trues=id2p_trues[d_id],
            corrects=id2corrects[d_id],
            alpha=alpha,
            beta=beta,
            rho_max=rho_max,
            tau=tau,
            var_norm=id2var_norm[d_id],
            use_confidence=use_confidence,
            use_variability=use_variability,
            use_forgetting=use_forgetting,
            conf_gate=conf_gate,
            conf_gamma=conf_gamma,
            conf_eps=conf_eps,
            return_details=True,
            conf_log_lambda=conf_log_lambda,
            lowconf_mask=bool(lowconf_mask_values[idx]),
            lowvar_tail_score=float(lowvar_tail_scores[idx]),
            lowvar_eta=lowvar_eta,
            not_highconf_mask=bool(not_highconf_mask_values[idx])
        )
        ref_loss_dic[d_id] = ref_loss
        for key in ['conf', 'var', 'var_norm', 'forg',
                    'conf_risk', 'var_risk', 'forg_risk', 'risk',
                    'rho', 'q_alpha', 'q_beta', 'ref_loss',
                    'lowvar_tail_residual', 'protected_lowvar_residual',
                    'gate_llr', 'gate_final']:
            diagnostics[key].append(details[key])
        diagnostics['final risk'].append(details['risk'])
    return ref_loss_dic, diagnostics


def select_by_loss_diff(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda, loss_params,
                        class_sizes=None):
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'], mse_factor=loss_params['mse_factor'], reduction='none')
    loss_diffs = {}
    id2pos = {}
    id2logits = {}
    batch_ids = []
    batch_sps = []
    batch_labs = []
    batch_logits = []
    with torch.no_grad():
        for i, di in enumerate(rand_data):
            if len(di) == 4:
                d_id, sp, lab, logit = di
            else:
                d_id, sp, lab = di
                logit = None
            id2pos[d_id] = i
            if transforms is not None:
                aug_sp = torch.unsqueeze(transforms(sp), dim=0)
            else:
                aug_sp = torch.unsqueeze(sp, dim=0)
            batch_ids.append(d_id)
            batch_sps.append(aug_sp)
            batch_labs.append(int(lab))
            if logit is not None:
                batch_logits.append(
                    torch.unsqueeze(torch.tensor(logit, dtype=torch.float32), dim=0)
                )
            if i % 32 == 0 or i == len(rand_data) - 1:
                sps = torch.cat(batch_sps, dim=0)
                labs = torch.tensor(batch_labs, dtype=torch.long)
                if len(batch_logits) > 0:
                    lab_logits = torch.cat(batch_logits, dim=0)
                else:
                    lab_logits = None
                if on_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()
                    if lab_logits is not None:
                        lab_logits = lab_logits.cuda()
                loss = loss_fn(x=model(sps), y=labs, logits=lab_logits)
                loss = loss.clone().detach()
                if on_cuda:
                    loss = loss.cpu()
                loss = loss.numpy()
                if lab_logits is not None:
                    if on_cuda:
                        lab_logits = lab_logits.cpu()
                    lab_logits = lab_logits.clone().detach().numpy()
                for j in range(len(batch_labs)):
                    did = batch_ids[j]
                    loss_dif = float(loss[j] - ref_loss_dic[did])
                    loss_diffs[did] = loss_dif
                    if lab_logits is not None:
                        id2logits[did] = lab_logits[j, :]
                batch_ids.clear()
                batch_sps.clear()
                batch_labs.clear()
                batch_logits.clear()
                del lab_logits
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    selected_data = []
    id2loss_dif = {}
    class_cnt = {}
    if class_sizes is not None:
        for ci in class_sizes.keys():
            class_cnt[ci] = 0
    for i in range(len(sorted_loss_diffs)):
        d_id = sorted_loss_diffs[i][0]
        pos = id2pos[d_id]
        di = rand_data[pos]
        if class_sizes is not None:
            lab = int(di[2])
            if class_cnt[lab] == class_sizes[lab]:
                continue
            else:
                class_cnt[lab] += 1
        new_di = copy.deepcopy(di)
        if loss_params['mse_factor'] > 0 and len(di) < 4:
            new_di.append(id2logits[d_id])
        selected_data.append(new_di)
        id2loss_dif[d_id] = sorted_loss_diffs[i][1]
        if len(selected_data) == incremental_size:
            break
    if on_cuda:
        model.cpu()
    model.train(status)
    return selected_data, id2loss_dif


def _model_features(model, x):
    fallback_to_logits = False
    if hasattr(model, 'forward_features'):
        feat = model.forward_features(x)
    else:
        try:
            feat = model(x, return_features=True)
        except TypeError:
            if hasattr(model, 'features') and callable(model.features):
                feat = model.features(x)
            elif hasattr(model, 'embed') and callable(model.embed):
                feat = model.embed(x)
            else:
                try:
                    feat = model(x, returnt='features')
                except TypeError:
                    feat = model(x)
                    fallback_to_logits = True
    if isinstance(feat, tuple):
        feat = feat[-1]
    if isinstance(feat, list):
        feat = feat[-1]
    feat = feat.view(feat.size(0), -1)
    return feat, fallback_to_logits


def _extract_candidate_features(rand_data, candidate_ids, id2pos, model, transforms, on_cuda):
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()
    features = []
    fallback_to_logits = False
    batch_sps = []
    try:
        with torch.no_grad():
            for i, d_id in enumerate(candidate_ids):
                di = rand_data[id2pos[d_id]]
                sp = di[1]
                if transforms is not None:
                    aug_sp = torch.unsqueeze(transforms(sp), dim=0)
                else:
                    aug_sp = torch.unsqueeze(sp, dim=0)
                batch_sps.append(aug_sp)
                if len(batch_sps) == 32 or i == len(candidate_ids) - 1:
                    sps = torch.cat(batch_sps, dim=0)
                    if on_cuda:
                        sps = sps.cuda()
                    feat, used_logits = _model_features(model, sps)
                    fallback_to_logits = fallback_to_logits or used_logits
                    if on_cuda:
                        feat = feat.cpu()
                    features.append(feat.clone().detach())
                    batch_sps.clear()
    finally:
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(random_rng_state)
        if on_cuda:
            model.cpu()
        model.train(status)
    features = torch.cat(features, dim=0)
    features = torch.nn.functional.normalize(features, p=2, dim=1, eps=1e-12)
    return features, fallback_to_logits


def _loss_diff_for_candidates(ref_loss_dic, rand_data, model, transforms, on_cuda, loss_params):
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'], mse_factor=loss_params['mse_factor'], reduction='none')
    loss_diffs = {}
    id2pos = {}
    id2logits = {}
    batch_ids = []
    batch_sps = []
    batch_labs = []
    batch_logits = []
    with torch.no_grad():
        for i, di in enumerate(rand_data):
            if len(di) == 4:
                d_id, sp, lab, logit = di
            else:
                d_id, sp, lab = di
                logit = None
            id2pos[d_id] = i
            if transforms is not None:
                aug_sp = torch.unsqueeze(transforms(sp), dim=0)
            else:
                aug_sp = torch.unsqueeze(sp, dim=0)
            batch_ids.append(d_id)
            batch_sps.append(aug_sp)
            batch_labs.append(int(lab))
            if logit is not None:
                batch_logits.append(
                    torch.unsqueeze(torch.tensor(logit, dtype=torch.float32), dim=0)
                )
            if i % 32 == 0 or i == len(rand_data) - 1:
                sps = torch.cat(batch_sps, dim=0)
                labs = torch.tensor(batch_labs, dtype=torch.long)
                if len(batch_logits) > 0:
                    lab_logits = torch.cat(batch_logits, dim=0)
                else:
                    lab_logits = None
                if on_cuda:
                    sps = sps.cuda()
                    labs = labs.cuda()
                    if lab_logits is not None:
                        lab_logits = lab_logits.cuda()
                loss = loss_fn(x=model(sps), y=labs, logits=lab_logits)
                loss = loss.clone().detach()
                if on_cuda:
                    loss = loss.cpu()
                loss = loss.numpy()
                if lab_logits is not None:
                    if on_cuda:
                        lab_logits = lab_logits.cpu()
                    lab_logits = lab_logits.clone().detach().numpy()
                for j in range(len(batch_labs)):
                    did = batch_ids[j]
                    loss_dif = float(loss[j] - ref_loss_dic[did])
                    loss_diffs[did] = loss_dif
                    if lab_logits is not None:
                        id2logits[did] = lab_logits[j, :]
                batch_ids.clear()
                batch_sps.clear()
                batch_labs.clear()
                batch_logits.clear()
                del lab_logits
    if on_cuda:
        model.cpu()
    model.train(status)
    return loss_diffs, id2pos, id2logits


def _make_selected_data(sorted_ids, rand_data, id2pos, id2logits, loss_params, class_sizes, incremental_size):
    selected_data = []
    id2loss_dif = {}
    class_cnt = {}
    if class_sizes is not None:
        for ci in class_sizes.keys():
            class_cnt[ci] = 0
    for d_id, loss_dif in sorted_ids:
        pos = id2pos[d_id]
        di = rand_data[pos]
        if class_sizes is not None:
            lab = int(di[2])
            if class_cnt[lab] == class_sizes[lab]:
                continue
            else:
                class_cnt[lab] += 1
        new_di = copy.deepcopy(di)
        if loss_params['mse_factor'] > 0 and len(di) < 4:
            new_di.append(id2logits[d_id])
        selected_data.append(new_di)
        id2loss_dif[d_id] = loss_dif
        if len(selected_data) == incremental_size:
            break
    return selected_data, id2loss_dif


def make_gss_anchor_replace_stats():
    return {
        'gss_anchor_replace_chunks_total': 0,
        'gss_anchor_replace_final_kept_total': 0,
        'gss_anchor_replace_tail_candidates_total': 0,
        'gss_anchor_replace_replaced_total': 0,
        'gss_anchor_replace_skip_similar_total': 0,
        'gss_anchor_replace_similarity_sum': 0.0,
        'gss_anchor_replace_similarity_count': 0,
        'gss_anchor_replace_similarity_max': 0.0,
        'gss_anchor_replace_rel_gap_sum': 0.0,
        'gss_anchor_replace_rel_gap_count': 0,
        'gss_anchor_replace_current_task_chunks': 0,
        'gss_anchor_replace_historical_task_chunks': 0,
        'gss_anchor_replace_current_anchor_replace_count': 0,
        'gss_anchor_replace_historical_anchor_replace_count': 0,
        'gss_anchor_replace_historical_top4_count': 0,
        'gss_anchor_replace_historical_fallback_count': 0,
        'gss_anchor_replace_permanent_rejected_total': 0,
        'gss_anchor_replace_grad_layer_used': '',
        'gss_anchor_replace_prob_candidate_total': 0,
        'gss_anchor_replace_prob_accept_total': 0,
        'gss_anchor_replace_prob_reject_total': 0,
        'gss_anchor_replace_prob_p_sum': 0.0,
        'gss_anchor_replace_prob_p_count': 0,
        'gss_anchor_replace_prob_p_min': None,
        'gss_anchor_replace_prob_p_max': None,
        'gss_anchor_replace_prob_target_sim_sum': 0.0,
        'gss_anchor_replace_prob_target_sim_count': 0,
        'gss_anchor_replace_prob_candidate_sim_sum': 0.0,
        'gss_anchor_replace_prob_candidate_sim_count': 0,
        'gss_anchor_replace_prob_u_sum': 0.0,
        'gss_anchor_replace_prob_u_count': 0,
        'gss_anchor_replace_prob_conservativeness': 1.0
    }


def make_gss_iqp_stats():
    return {
        'gss_iqp_block_id': 0,
        'gss_iqp_chunks_total': 0,
        'gss_iqp_final_kept_total': 0,
        'gss_iqp_current_iqp_chunks': 0,
        'gss_iqp_historical_top4_count': 0,
        'gss_iqp_historical_iqp_count': 0,
        'gss_iqp_changed_chunks': 0,
        'gss_iqp_changed_samples': 0,
        'gss_iqp_pairwise_cos_rel_top4_sum': 0.0,
        'gss_iqp_pairwise_cos_iqp_selected_sum': 0.0,
        'gss_iqp_rel_sum_rel_top4_sum': 0.0,
        'gss_iqp_rel_sum_iqp_selected_sum': 0.0,
        'gss_iqp_permanent_rejected_total': 0,
        'gss_iqp_grad_layer_used': ''
    }


def gss_anchor_replace_is_strategy(selection_strategy):
    return selection_strategy in [
        'rel_gss_anchor_replace',
        'rel_gss_anchor_replace_all_tasks',
        'rel_gss_anchor_replace_hist_top4',
        'rel_gss_anchor_replace_hist_top4_prob'
    ]


def gss_anchor_replace_uses_anchor(selection_strategy, is_current_task):
    if selection_strategy == 'rel_gss_anchor_replace_all_tasks':
        return True
    return selection_strategy in [
        'rel_gss_anchor_replace',
        'rel_gss_anchor_replace_hist_top4',
        'rel_gss_anchor_replace_hist_top4_prob'
    ] and bool(is_current_task)


def gss_anchor_replace_uses_historical_rel_top4(selection_strategy, is_current_task):
    return selection_strategy in [
        'rel_gss_anchor_replace_hist_top4',
        'rel_gss_anchor_replace_hist_top4_prob'
    ] and not bool(is_current_task)


def gss_anchor_replace_uses_prob_replace(selection_strategy):
    return selection_strategy == 'rel_gss_anchor_replace_hist_top4_prob'


def gss_iqp_is_strategy(selection_strategy):
    return selection_strategy == 'rel_gss_iqp_hist_top4'


def gss_iqp_uses_iqp(selection_strategy, is_current_task):
    return selection_strategy == 'rel_gss_iqp_hist_top4' and bool(is_current_task)


def gss_iqp_uses_historical_rel_top4(selection_strategy, is_current_task):
    return selection_strategy == 'rel_gss_iqp_hist_top4' and not bool(is_current_task)


def gss_anchor_replace_resolve_chunk_size(selection_strategy, is_current_task, selection_chunk_size,
                                          base_incremental_size, remaining_select_size):
    if selection_strategy == 'rel_gss_anchor_replace' and not bool(is_current_task):
        return 1
    if selection_chunk_size > 0:
        return min(selection_chunk_size, remaining_select_size)
    return min(base_incremental_size, remaining_select_size)


def _add_gss_anchor_replace_stat(stats, key, value):
    if stats is not None:
        stats[key] = stats.get(key, 0) + value


def _set_gss_anchor_replace_stat(stats, key, value):
    if stats is not None:
        stats[key] = value


def _gss_anchor_replace_rng_uniform(prob_rng):
    if hasattr(prob_rng, 'random_sample'):
        return float(prob_rng.random_sample())
    if hasattr(prob_rng, 'random'):
        return float(prob_rng.random())
    raise ValueError('Invalid probabilistic replacement RNG')


def compute_gss_anchor_replace_prob(target_sim, candidate_sim, conservativeness=1.0):
    conservativeness = float(conservativeness)
    if conservativeness <= 0.0:
        raise ValueError('gss_anchor_replace_prob_conservativeness must be > 0')
    target_score = float(target_sim) + 1.0
    candidate_score = float(candidate_sim) + 1.0
    denom = target_score + conservativeness * candidate_score
    if denom <= 1e-12:
        return 0.5
    return min(max(float(target_score / denom), 0.0), 1.0)


def _record_gss_anchor_replace_prob(stats, candidate_sim, target_sim, p_replace, u_value, accepted):
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_candidate_total', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_p_sum', float(p_replace))
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_p_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_target_sim_sum', float(target_sim))
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_target_sim_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_candidate_sim_sum', float(candidate_sim))
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_candidate_sim_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_u_sum', float(u_value))
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_u_count', 1)
    if bool(accepted):
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_accept_total', 1)
    else:
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_prob_reject_total', 1)
    if stats is not None:
        old_p_min = stats.get('gss_anchor_replace_prob_p_min', None)
        old_p_max = stats.get('gss_anchor_replace_prob_p_max', None)
        stats['gss_anchor_replace_prob_p_min'] = (
            float(p_replace) if old_p_min is None else min(float(old_p_min), float(p_replace)))
        stats['gss_anchor_replace_prob_p_max'] = (
            float(p_replace) if old_p_max is None else max(float(old_p_max), float(p_replace)))


def record_gss_anchor_replace_historical_top4(stats, final_kept):
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_chunks_total', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_historical_task_chunks', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_historical_top4_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_final_kept_total', int(final_kept))


def record_gss_iqp_historical_top4(stats, final_kept):
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_chunks_total', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_historical_top4_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_final_kept_total', int(final_kept))


def _normalize_grad_vector(grad):
    grad = grad.view(-1).float()
    return torch.nn.functional.normalize(grad, p=2, dim=0, eps=1e-12)


def _resolve_grad_parameters(model, grad_layer):
    requested_layer = str(grad_layer)
    direct_names = [requested_layer]
    if requested_layer == 'classifier':
        direct_names = ['classifier', 'linear', 'fc', 'fc3', 'fc2', 'head']
    for module_name in direct_names:
        if hasattr(model, module_name):
            module = getattr(model, module_name)
            params = [(module_name + '.' + name, param)
                      for name, param in module.named_parameters(recurse=True)
                      if param.requires_grad]
            if len(params) > 0:
                return params, module_name
    for module_name, module in model.named_modules():
        if module_name == requested_layer:
            params = [(module_name + '.' + name, param)
                      for name, param in module.named_parameters(recurse=True)
                      if param.requires_grad]
            if len(params) > 0:
                return params, module_name
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad and (name == requested_layer or name.startswith(requested_layer + '.')):
            params.append((name, param))
    if len(params) > 0:
        return params, requested_layer
    if requested_layer == 'classifier':
        classifier_tokens = ['classifier', 'linear', 'fc', 'head']
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if any(token in name for token in classifier_tokens):
                params.append((name, param))
        if len(params) > 0:
            return params, 'classifier_auto'
    params = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    return params, 'all'


def _extract_candidate_gradients(rand_data, candidate_ids, id2pos, model, transforms, on_cuda, loss_params,
                                 grad_layer):
    status = model.training
    model.eval()
    if on_cuda:
        model.cuda()
    grad_params, grad_layer_used = _resolve_grad_parameters(model=model, grad_layer=grad_layer)
    loss_fn = loss_functions.CompliedLoss(
        ce_factor=loss_params['ce_factor'], mse_factor=loss_params['mse_factor'], reduction='mean')
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    np_rng_state = np.random.get_state()
    random_rng_state = random.getstate()
    gradients_by_id = {}
    try:
        for d_id in candidate_ids:
            di = rand_data[id2pos[d_id]]
            if len(di) == 4:
                _, sp, lab, logit = di
            else:
                _, sp, lab = di
                logit = None
            if transforms is not None:
                sps = torch.unsqueeze(transforms(sp), dim=0)
            else:
                sps = torch.unsqueeze(sp, dim=0)
            labs = torch.tensor([int(lab)], dtype=torch.long)
            if logit is not None:
                lab_logits = torch.unsqueeze(torch.tensor(logit, dtype=torch.float32), dim=0)
            else:
                lab_logits = None
            if on_cuda:
                sps = sps.cuda()
                labs = labs.cuda()
                if lab_logits is not None:
                    lab_logits = lab_logits.cuda()
            model.zero_grad()
            loss = loss_fn(x=model(sps), y=labs, logits=lab_logits)
            if hasattr(loss, 'dim') and loss.dim() > 0:
                loss = loss.mean()
            loss.backward()
            grad_parts = []
            for _, param in grad_params:
                if param.grad is None:
                    grad_parts.append(torch.zeros(param.numel(), dtype=torch.float32))
                else:
                    grad_parts.append(param.grad.detach().view(-1).cpu().float())
            if len(grad_parts) == 0:
                gradients_by_id[int(d_id)] = torch.zeros(1, dtype=torch.float32)
            else:
                gradients_by_id[int(d_id)] = torch.cat(grad_parts, dim=0)
            model.zero_grad()
    finally:
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        np.random.set_state(np_rng_state)
        random.setstate(random_rng_state)
        if on_cuda:
            model.cpu()
        model.train(status)
    return gradients_by_id, grad_layer_used


def _pairwise_cosine_sum(items, normalized_gradients):
    total = 0.0
    if len(items) < 2:
        return total
    for i, j in combinations(range(len(items)), 2):
        grad_i = normalized_gradients[int(items[i][0])]
        grad_j = normalized_gradients[int(items[j][0])]
        total += float(torch.dot(grad_i, grad_j))
    return total


def _rel_sum(items):
    return float(sum(float(item[1]) for item in items))


def _rank_sum(items, rank_by_id):
    return int(sum(rank_by_id[int(item[0])] for item in items))


def _stable_id_tuple(items):
    return tuple(int(item[0]) for item in items)


def _iqp_combo_is_better(pairwise_sum, rel_sum, rank_sum, id_tuple,
                         best_pairwise_sum, best_rel_sum, best_rank_sum, best_id_tuple):
    eps = 1e-12
    if best_pairwise_sum is None:
        return True
    if pairwise_sum < best_pairwise_sum - eps:
        return True
    if abs(pairwise_sum - best_pairwise_sum) > eps:
        return False
    if rel_sum > best_rel_sum + eps:
        return True
    if abs(rel_sum - best_rel_sum) > eps:
        return False
    if rank_sum < best_rank_sum:
        return True
    if rank_sum > best_rank_sum:
        return False
    return id_tuple < best_id_tuple


def select_gss_iqp_ranked_items(sorted_loss_diffs, gradients_by_id, window_size=8, select_size=4,
                                stats=None, is_current_task=True):
    window_size = max(int(window_size), 1)
    select_size = max(int(select_size), 1)
    window_size = max(window_size, select_size)
    window_items = list(sorted_loss_diffs[:min(window_size, len(sorted_loss_diffs))])
    rel_top_items = list(window_items[:min(select_size, len(window_items))])
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_chunks_total', 1)
    if not bool(is_current_task):
        _add_gss_anchor_replace_stat(stats, 'gss_iqp_historical_top4_count', 1)
        _add_gss_anchor_replace_stat(stats, 'gss_iqp_final_kept_total', len(rel_top_items))
        return rel_top_items
    if len(window_items) == 0:
        return []
    if len(window_items) < select_size:
        _add_gss_anchor_replace_stat(stats, 'gss_iqp_final_kept_total', len(window_items))
        return window_items
    normalized_gradients = {
        int(item[0]): _normalize_grad_vector(gradients_by_id[int(item[0])])
        for item in window_items
    }
    rel_top_pairwise = _pairwise_cosine_sum(rel_top_items, normalized_gradients)
    rel_top_rel_sum = _rel_sum(rel_top_items)
    selected_items = rel_top_items
    selected_pairwise = rel_top_pairwise
    selected_rel_sum = rel_top_rel_sum
    if len(window_items) >= select_size:
        _add_gss_anchor_replace_stat(stats, 'gss_iqp_current_iqp_chunks', 1)
        rank_by_id = {int(item[0]): rank for rank, item in enumerate(window_items)}
        best_combo = None
        best_pairwise_sum = None
        best_rel_sum = None
        best_rank_sum = None
        best_id_tuple = None
        for combo in combinations(window_items, select_size):
            combo = list(combo)
            combo_pairwise_sum = _pairwise_cosine_sum(combo, normalized_gradients)
            combo_rel_sum = _rel_sum(combo)
            combo_rank_sum = _rank_sum(combo, rank_by_id)
            combo_id_tuple = _stable_id_tuple(combo)
            if _iqp_combo_is_better(combo_pairwise_sum, combo_rel_sum, combo_rank_sum, combo_id_tuple,
                                    best_pairwise_sum, best_rel_sum, best_rank_sum, best_id_tuple):
                best_combo = combo
                best_pairwise_sum = combo_pairwise_sum
                best_rel_sum = combo_rel_sum
                best_rank_sum = combo_rank_sum
                best_id_tuple = combo_id_tuple
        selected_items = best_combo
        selected_pairwise = best_pairwise_sum
        selected_rel_sum = best_rel_sum
    rel_ids = set(int(item[0]) for item in rel_top_items)
    selected_ids = set(int(item[0]) for item in selected_items)
    if rel_ids != selected_ids:
        _add_gss_anchor_replace_stat(stats, 'gss_iqp_changed_chunks', 1)
        _add_gss_anchor_replace_stat(stats, 'gss_iqp_changed_samples', len(rel_ids - selected_ids))
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_pairwise_cos_rel_top4_sum', rel_top_pairwise)
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_pairwise_cos_iqp_selected_sum', selected_pairwise)
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_rel_sum_rel_top4_sum', rel_top_rel_sum)
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_rel_sum_iqp_selected_sum', selected_rel_sum)
    _add_gss_anchor_replace_stat(stats, 'gss_iqp_final_kept_total', len(selected_items))
    return selected_items


def select_anchor_replace_ranked_items(sorted_loss_diffs, gradients_by_id, window_size=8, anchor_size=4,
                                       sim_threshold=0.90, stats=None, is_current_task=True,
                                       probabilistic_replace=False, prob_rng=None,
                                       prob_conservativeness=1.0):
    window_size = max(int(window_size), 1)
    anchor_size = max(int(anchor_size), 1)
    anchor_size = min(anchor_size, len(sorted_loss_diffs))
    window_size = max(window_size, anchor_size)
    window_items = list(sorted_loss_diffs[:min(window_size, len(sorted_loss_diffs))])
    selected_items = list(window_items[:anchor_size])
    tail_items = list(window_items[anchor_size:])
    normalized_gradients = {
        int(item[0]): _normalize_grad_vector(gradients_by_id[int(item[0])])
        for item in window_items
    }
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_chunks_total', 1)
    if bool(is_current_task):
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_current_task_chunks', 1)
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_current_anchor_replace_count', 1)
    else:
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_historical_task_chunks', 1)
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_historical_anchor_replace_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_tail_candidates_total', len(tail_items))
    if bool(probabilistic_replace):
        prob_conservativeness = float(prob_conservativeness)
        if prob_conservativeness <= 0.0:
            raise ValueError('gss_anchor_replace_prob_conservativeness must be > 0')
        _set_gss_anchor_replace_stat(
            stats, 'gss_anchor_replace_prob_conservativeness', prob_conservativeness)
    for candidate_item in tail_items:
        cand_id, cand_rel = candidate_item
        cand_grad = normalized_gradients[int(cand_id)]
        sim_values = []
        for selected_item in selected_items:
            selected_id = int(selected_item[0])
            selected_grad = normalized_gradients[selected_id]
            sim_value = float(torch.dot(cand_grad, selected_grad))
            sim_values.append(sim_value)
        sim_max = max(sim_values) if len(sim_values) > 0 else 0.0
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_similarity_sum', sim_max)
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_similarity_count', 1)
        if stats is not None:
            stats['gss_anchor_replace_similarity_max'] = max(
                stats.get('gss_anchor_replace_similarity_max', 0.0), sim_max)
        if sim_max >= float(sim_threshold):
            _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_skip_similar_total', 1)
            continue
        replace_pos = min(range(len(selected_items)), key=lambda idx: float(selected_items[idx][1]))
        replace_target = selected_items[replace_pos]
        if bool(probabilistic_replace):
            if prob_rng is None:
                raise ValueError('prob_rng is required when probabilistic_replace=True')
            target_id = int(replace_target[0])
            target_grad = normalized_gradients[target_id]
            target_sim_values = []
            for idx, selected_item in enumerate(selected_items):
                if idx == replace_pos:
                    continue
                selected_id = int(selected_item[0])
                selected_grad = normalized_gradients[selected_id]
                target_sim_values.append(float(torch.dot(target_grad, selected_grad)))
            target_sim = max(target_sim_values) if len(target_sim_values) > 0 else 0.0
            p_replace = compute_gss_anchor_replace_prob(
                target_sim=target_sim,
                candidate_sim=sim_max,
                conservativeness=prob_conservativeness
            )
            u_value = _gss_anchor_replace_rng_uniform(prob_rng)
            accepted = u_value < p_replace
            _record_gss_anchor_replace_prob(
                stats=stats,
                candidate_sim=sim_max,
                target_sim=target_sim,
                p_replace=p_replace,
                u_value=u_value,
                accepted=accepted
            )
            if not bool(accepted):
                continue
        selected_items[replace_pos] = candidate_item
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_replaced_total', 1)
        _add_gss_anchor_replace_stat(
            stats, 'gss_anchor_replace_rel_gap_sum', float(replace_target[1]) - float(cand_rel))
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_rel_gap_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_final_kept_total', len(selected_items))
    return selected_items


def select_by_loss_diff_with_gss_iqp(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda,
                                     loss_params, class_sizes=None, window_size=8, select_size=4,
                                     grad_layer='classifier', stats=None, is_current_task=True):
    loss_diffs, id2pos, id2logits = _loss_diff_for_candidates(
        ref_loss_dic=ref_loss_dic,
        rand_data=rand_data,
        model=model,
        transforms=transforms,
        on_cuda=on_cuda,
        loss_params=loss_params
    )
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    select_size = min(int(select_size), int(incremental_size))
    window_size = max(int(window_size), select_size)
    window_items = sorted_loss_diffs[:min(window_size, len(sorted_loss_diffs))]
    candidate_ids = [x[0] for x in window_items]
    gradients_by_id, grad_layer_used = _extract_candidate_gradients(
        rand_data=rand_data,
        candidate_ids=candidate_ids,
        id2pos=id2pos,
        model=model,
        transforms=transforms,
        on_cuda=on_cuda,
        loss_params=loss_params,
        grad_layer=grad_layer
    )
    _set_gss_anchor_replace_stat(stats, 'gss_iqp_grad_layer_used', grad_layer_used)
    selected_items = select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=gradients_by_id,
        window_size=window_size,
        select_size=select_size,
        stats=stats,
        is_current_task=is_current_task
    )
    selected_data, id2loss_dif = _make_selected_data(
        sorted_ids=selected_items,
        rand_data=rand_data,
        id2pos=id2pos,
        id2logits=id2logits,
        loss_params=loss_params,
        class_sizes=class_sizes,
        incremental_size=len(selected_items)
    )
    _print_selected_data_checks(selected_data, len(selected_items))
    return selected_data, id2loss_dif


def select_by_loss_diff_with_anchor_replace(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda,
                                            loss_params, class_sizes=None, window_size=8, anchor_size=4,
                                            sim_threshold=0.90, grad_layer='classifier', stats=None,
                                            is_current_task=True, probabilistic_replace=False, prob_rng=None,
                                            prob_conservativeness=1.0):
    loss_diffs, id2pos, id2logits = _loss_diff_for_candidates(
        ref_loss_dic=ref_loss_dic,
        rand_data=rand_data,
        model=model,
        transforms=transforms,
        on_cuda=on_cuda,
        loss_params=loss_params
    )
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    anchor_size = min(int(anchor_size), int(incremental_size), len(sorted_loss_diffs))
    window_size = max(int(window_size), anchor_size)
    window_items = sorted_loss_diffs[:min(window_size, len(sorted_loss_diffs))]
    candidate_ids = [x[0] for x in window_items]
    gradients_by_id, grad_layer_used = _extract_candidate_gradients(
        rand_data=rand_data,
        candidate_ids=candidate_ids,
        id2pos=id2pos,
        model=model,
        transforms=transforms,
        on_cuda=on_cuda,
        loss_params=loss_params,
        grad_layer=grad_layer
    )
    _set_gss_anchor_replace_stat(stats, 'gss_anchor_replace_grad_layer_used', grad_layer_used)
    selected_items = select_anchor_replace_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=gradients_by_id,
        window_size=window_size,
        anchor_size=anchor_size,
        sim_threshold=sim_threshold,
        stats=stats,
        is_current_task=is_current_task,
        probabilistic_replace=probabilistic_replace,
        prob_rng=prob_rng,
        prob_conservativeness=prob_conservativeness
    )
    selected_data, id2loss_dif = _make_selected_data(
        sorted_ids=selected_items,
        rand_data=rand_data,
        id2pos=id2pos,
        id2logits=id2logits,
        loss_params=loss_params,
        class_sizes=class_sizes,
        incremental_size=len(selected_items)
    )
    _print_selected_data_checks(selected_data, len(selected_items))
    return selected_data, id2loss_dif


def select_by_loss_diff_with_diversity(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda,
                                       loss_params, class_sizes=None, feature_model=None, div_lambda=0.1,
                                       div_candidate_ratio=3, div_feature_source='current_model',
                                       div_feature_layer='penultimate', div_verbose=False,
                                       selected_features_bank=None, selected_ids_bank=None, stateful_diag=None):
    loss_diffs, id2pos, id2logits = _loss_diff_for_candidates(
        ref_loss_dic=ref_loss_dic,
        rand_data=rand_data,
        model=model,
        transforms=transforms,
        on_cuda=on_cuda,
        loss_params=loss_params
    )
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    stateful_rd = selected_features_bank is not None and selected_ids_bank is not None
    if stateful_diag is not None:
        stateful_diag['calls'] = stateful_diag.get('calls', 0) + 1
        stateful_diag.setdefault('selection_sizes', []).append(incremental_size)
    if div_lambda == 0 or (incremental_size <= 1 and not stateful_rd):
        selected_data, id2loss_dif = _make_selected_data(
            sorted_ids=sorted_loss_diffs,
            rand_data=rand_data,
            id2pos=id2pos,
            id2logits=id2logits,
            loss_params=loss_params,
            class_sizes=class_sizes,
            incremental_size=incremental_size
        )
        rel_ids = [int(x[0]) for x in sorted_loss_diffs[:incremental_size]]
        lambda0_ids = [int(x[0]) for x in selected_data]
        overlap = len(set(rel_ids).intersection(set(lambda0_ids))) / max(len(rel_ids), 1)
        if div_lambda == 0:
            print('[RD-CHECK] lambda0_overlap_with_rel=%.6f' % overlap)
            print('[RD-CHECK] lambda0_ids_equal=' + str(rel_ids == lambda0_ids))
        if div_verbose and incremental_size <= 1 and div_lambda != 0:
            print('[REL-DIVERSITY] selection_size=' + str(incremental_size))
            print('[REL-DIVERSITY] candidate_size=' + str(min(len(sorted_loss_diffs), int(incremental_size * div_candidate_ratio))))
            print('[REL-DIVERSITY] div_lambda=' + str(div_lambda))
            print('[REL-DIVERSITY] div_candidate_ratio=' + str(div_candidate_ratio))
            print('[REL-DIVERSITY] feature_source=' + str(div_feature_source))
            print('[REL-DIVERSITY] feature_layer=' + str(div_feature_layer))
            print('[REL-DIVERSITY] fallback_to_logits=False')
            print('[REL-DIVERSITY] rel_norm_mean=0.000000')
            print('[REL-DIVERSITY] rel_norm_std=0.000000')
            print('[REL-DIVERSITY] selected_rel_mean=%.6f' % float(np.mean([x[1] for x in sorted_loss_diffs[:incremental_size]])))
            print('[REL-DIVERSITY] selected_max_similarity_mean=0.000000')
            print('[REL-DIVERSITY] selected_score_mean=%.6f' % float(np.mean([x[1] for x in sorted_loss_diffs[:incremental_size]])))
            print('[REL-DIVERSITY] selected_id_count=' + str(len(selected_data)))
            print('[REL-DIVERSITY] duplicate_id_count=0')
        _print_selected_data_checks(selected_data, incremental_size)
        return selected_data, id2loss_dif
    candidate_size = min(len(sorted_loss_diffs), int(incremental_size * div_candidate_ratio))
    candidate_size = max(candidate_size, incremental_size)
    candidate_items = sorted_loss_diffs[:candidate_size]
    candidate_ids = [x[0] for x in candidate_items]
    rel = torch.tensor([x[1] for x in candidate_items], dtype=torch.float32)
    rel_mean = rel.mean()
    rel_std = rel.std(unbiased=False)
    rel_norm = (rel - rel_mean) / (rel_std + 1e-8)
    if feature_model is None:
        feature_model = model
    features, fallback_to_logits = _extract_candidate_features(
        rand_data=rand_data,
        candidate_ids=candidate_ids,
        id2pos=id2pos,
        model=feature_model,
        transforms=transforms,
        on_cuda=on_cuda
    )
    bank_size_before = len(selected_features_bank) if stateful_rd else 0
    selected_pos = []
    remaining = set(range(candidate_size))
    selected_scores = []
    selected_max_sims = []
    while len(selected_pos) < incremental_size and len(remaining) > 0:
        rem_pos = sorted(list(remaining))
        rem_tensor = torch.tensor(rem_pos, dtype=torch.long)
        compare_features = []
        if stateful_rd and len(selected_features_bank) > 0:
            compare_features.append(torch.stack(selected_features_bank, dim=0))
        if len(selected_pos) > 0:
            sel_tensor = torch.tensor(selected_pos, dtype=torch.long)
            compare_features.append(features[sel_tensor])
        if len(compare_features) > 0:
            selected_features = torch.cat(compare_features, dim=0)
            sims = torch.mm(features[rem_tensor], selected_features.t())
            max_sims, _ = torch.max(sims, dim=1)
        else:
            max_sims = torch.zeros(len(rem_pos), dtype=torch.float32)
        scores = rel_norm[rem_tensor] - float(div_lambda) * max_sims
        best_local = int(torch.argmax(scores).item())
        best_pos = rem_pos[best_local]
        selected_pos.append(best_pos)
        remaining.remove(best_pos)
        selected_scores.append(float(scores[best_local]))
        selected_max_sims.append(float(max_sims[best_local]))
    selected_items = [(candidate_ids[pos], float(rel[pos])) for pos in selected_pos]
    selected_data, id2loss_dif = _make_selected_data(
        sorted_ids=selected_items,
        rand_data=rand_data,
        id2pos=id2pos,
        id2logits=id2logits,
        loss_params=loss_params,
        class_sizes=class_sizes,
        incremental_size=incremental_size
    )
    if stateful_rd:
        id2candidate_pos = {}
        for pos, d_id in enumerate(candidate_ids):
            id2candidate_pos[int(d_id)] = pos
        for di in selected_data:
            d_id = int(di[0])
            if d_id in id2candidate_pos and d_id not in selected_ids_bank:
                selected_features_bank.append(features[id2candidate_pos[d_id]].clone().detach().cpu())
                selected_ids_bank.append(d_id)
    if stateful_diag is not None and bank_size_before > 0:
        if any(abs(x) > 1e-8 for x in selected_max_sims):
            stateful_diag['max_similarity_nonzero_after_first'] = True
        if float(div_lambda) != 0.0 and any(abs(float(div_lambda) * x) > 1e-8 for x in selected_max_sims):
            stateful_diag['div_penalty_active'] = True
    if div_verbose:
        duplicate_id_count = len(selected_data) - len(set([int(x[0]) for x in selected_data]))
        selected_rel = [float(id2loss_dif[int(x[0])]) for x in selected_data]
        print('[REL-DIVERSITY] selection_size=' + str(incremental_size))
        print('[REL-DIVERSITY] candidate_size=' + str(candidate_size))
        print('[REL-DIVERSITY] div_lambda=' + str(div_lambda))
        print('[REL-DIVERSITY] div_candidate_ratio=' + str(div_candidate_ratio))
        print('[REL-DIVERSITY] feature_source=' + str(div_feature_source))
        print('[REL-DIVERSITY] feature_layer=' + str(div_feature_layer))
        print('[REL-DIVERSITY] fallback_to_logits=' + str(fallback_to_logits))
        print('[REL-DIVERSITY] rel_norm_mean=%.6f' % float(rel_norm.mean()))
        print('[REL-DIVERSITY] rel_norm_std=%.6f' % float(rel_norm.std(unbiased=False)))
        print('[REL-DIVERSITY] selected_rel_mean=%.6f' % float(np.mean(selected_rel)))
        print('[REL-DIVERSITY] selected_max_similarity_mean=%.6f' % float(np.mean(selected_max_sims)))
        print('[REL-DIVERSITY] selected_score_mean=%.6f' % float(np.mean(selected_scores)))
        print('[REL-DIVERSITY] selected_id_count=' + str(len(selected_data)))
        print('[REL-DIVERSITY] duplicate_id_count=' + str(duplicate_id_count))
        if fallback_to_logits:
            print('[REL-DIVERSITY] fallback_to_logits=True')
    _print_selected_data_checks(selected_data, incremental_size)
    return selected_data, id2loss_dif


def select_by_loss_diff_with_filter(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda,
                                    loss_params, class_sizes=None, feature_model=None, filter_sim_threshold=0.85,
                                    filter_feature_source='current_model', filter_feature_layer='penultimate',
                                    filter_reject_dominated=False, filter_verbose=False,
                                    rejected_total_before=0, max_rejections_this_chunk=None):
    loss_diffs, id2pos, id2logits = _loss_diff_for_candidates(
        ref_loss_dic=ref_loss_dic,
        rand_data=rand_data,
        model=model,
        transforms=transforms,
        on_cuda=on_cuda,
        loss_params=loss_params
    )
    sorted_loss_diffs = sorted(loss_diffs.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_loss_diffs[:incremental_size]
    candidate_ids = [x[0] for x in top_items]
    if feature_model is None:
        feature_model = model
    features, fallback_to_logits = _extract_candidate_features(
        rand_data=rand_data,
        candidate_ids=candidate_ids,
        id2pos=id2pos,
        model=feature_model,
        transforms=transforms,
        on_cuda=on_cuda
    )
    rel_by_id = {}
    rank_by_id = {}
    for rank, item in enumerate(top_items):
        d_id, loss_dif = item
        rel_by_id[int(d_id)] = float(loss_dif)
        rank_by_id[int(d_id)] = rank
    pairwise_values = []
    rejected_this_chunk = set()
    if len(candidate_ids) > 1:
        sims = torch.mm(features, features.t())
        for i in range(len(candidate_ids)):
            for j in range(i + 1, len(candidate_ids)):
                sim_value = float(sims[i, j])
                pairwise_values.append(sim_value)
                if filter_reject_dominated and sim_value > filter_sim_threshold:
                    id_i = int(candidate_ids[i])
                    id_j = int(candidate_ids[j])
                    rel_i = rel_by_id[id_i]
                    rel_j = rel_by_id[id_j]
                    if rel_i > rel_j:
                        rejected_this_chunk.add(id_j)
                    elif rel_j > rel_i:
                        rejected_this_chunk.add(id_i)
                    else:
                        if rank_by_id[id_i] < rank_by_id[id_j] or \
                                (rank_by_id[id_i] == rank_by_id[id_j] and id_i <= id_j):
                            rejected_this_chunk.add(id_j)
                        else:
                            rejected_this_chunk.add(id_i)
    selected_items = []
    if max_rejections_this_chunk is not None and len(rejected_this_chunk) > max_rejections_this_chunk:
        rejected_this_chunk = set(sorted(
            list(rejected_this_chunk),
            key=lambda x: (rel_by_id[int(x)], -rank_by_id[int(x)], int(x))
        )[:max_rejections_this_chunk])
    for d_id, loss_dif in top_items:
        if int(d_id) not in rejected_this_chunk:
            selected_items.append((d_id, loss_dif))
    if len(selected_items) == 0 and len(top_items) > 0:
        fallback_id = int(top_items[0][0])
        rejected_this_chunk.discard(fallback_id)
        selected_items.append(top_items[0])
    selected_data, id2loss_dif = _make_selected_data(
        sorted_ids=selected_items,
        rand_data=rand_data,
        id2pos=id2pos,
        id2logits=id2logits,
        loss_params=loss_params,
        class_sizes=class_sizes,
        incremental_size=len(selected_items)
    )
    selected_ids = set([int(di[0]) for di in selected_data])
    rejected_this_chunk = set([d_id for d_id in rejected_this_chunk if d_id not in selected_ids])
    rejected_ids = list(rejected_this_chunk)
    selected_rel = [float(id2loss_dif[int(x[0])]) for x in selected_data]
    rejected_rel = [rel_by_id[int(d_id)] for d_id in rejected_ids if int(d_id) in rel_by_id]
    rejected_intersection_selected_count = len(set(rejected_ids).intersection(selected_ids))
    if len(pairwise_values) > 0:
        pairwise_mean = float(np.mean(pairwise_values))
        pairwise_max = float(np.max(pairwise_values))
    else:
        pairwise_mean = 0.0
        pairwise_max = 0.0
    if filter_verbose:
        print('[REL-FILTER] chunk_size=' + str(incremental_size))
        print('[REL-FILTER] candidate_topk=' + str(len(candidate_ids)))
        print('[REL-FILTER] sim_threshold=' + str(filter_sim_threshold))
        print('[REL-FILTER] filter_feature_source=' + str(filter_feature_source))
        print('[REL-FILTER] filter_feature_layer=' + str(filter_feature_layer))
        print('[REL-FILTER] fallback_to_logits=' + str(fallback_to_logits))
        print('[REL-FILTER] top4_pairwise_similarity_mean=%.6f' % pairwise_mean)
        print('[REL-FILTER] top4_pairwise_similarity_max=%.6f' % pairwise_max)
        print('[REL-FILTER] rejected_this_chunk=' + str(len(rejected_ids)))
        print('[REL-FILTER] selected_this_chunk=' + str(len(selected_data)))
        print('[REL-FILTER] rejected_total=' + str(rejected_total_before + len(rejected_ids)))
        print('[REL-FILTER] selected_rel_mean=%.6f' %
              (float(np.mean(selected_rel)) if len(selected_rel) > 0 else 0.0))
        print('[REL-FILTER] rejected_rel_mean=%.6f' %
              (float(np.mean(rejected_rel)) if len(rejected_rel) > 0 else 0.0))
        print('[REL-FILTER] all_top4_selected=' + str(len(rejected_ids) == 0))
        print('[RF-CHECK] selected_data_structure_ok=' + str(_selected_data_structure_ok(selected_data)))
        selected_id_list = [int(di[0]) for di in selected_data]
        print('[RF-CHECK] duplicate_selected_id_count=' +
              str(len(selected_id_list) - len(set(selected_id_list))))
        print('[RF-CHECK] rejected_intersection_selected_count=' +
              str(rejected_intersection_selected_count))
        print('[RF-CHECK] rejected_scope=current_incremental_selection')
        if fallback_to_logits:
            print('[REL-FILTER] fallback_to_logits=True')
    return selected_data, id2loss_dif, rejected_ids


def _selected_data_structure_ok(selected_data):
    for di in selected_data:
        if not isinstance(di, list) or len(di) not in [3, 4]:
            return False
        if di[0] is None or di[1] is None or di[2] is None:
            return False
    return True


def _print_selected_data_checks(selected_data, incremental_size):
    len_counts = {}
    ids = []
    bad_item_count = 0
    for di in selected_data:
        if not isinstance(di, list):
            bad_item_count += 1
            continue
        len_counts[len(di)] = len_counts.get(len(di), 0) + 1
        if len(di) not in [3, 4]:
            bad_item_count += 1
        elif di[0] is None or di[1] is None or di[2] is None:
            bad_item_count += 1
        else:
            ids.append(int(di[0]))
    duplicate_id_count = len(ids) - len(set(ids))
    print('[RD-CHECK] selected_size_ok=' + str(len(selected_data) == incremental_size))
    print('[RD-CHECK] selected_item_len_counts=' + str(len_counts))
    print('[RD-CHECK] duplicate_id_count=' + str(duplicate_id_count))
    print('[RD-CHECK] bad_item_count=' + str(bad_item_count))
