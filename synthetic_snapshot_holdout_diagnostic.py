# -*-coding:utf8-*-

import argparse
import inspect
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch

import offline_continual_learning
import utils
from coreset_selection import coreset_selection_functions as funcs
from coreset_selection import selection_agent


MODEL_PARAMS = {
    'model_type': 'mlp',
    'input_dim': 2,
    'interm_dim': 4,
    'num_class': 2
}


def _make_opts():
    return argparse.Namespace(
        cur_train_steps=1,
        cur_train_lr=0.1,
        selection_steps=1,
        slt_mse_factor=0.0,
        selection_strategy='rel',
        selection_chunk_size=0,
        gss_anchor_replace_window=8,
        gss_anchor_replace_anchor_size=4,
        gss_anchor_replace_sim_threshold=0.90,
        gss_anchor_replace_prob_seed_offset=0,
        gss_anchor_replace_prob_conservativeness=1.0,
        gss_grad_layer='classifier',
        rel_ref_mode='trained_ref',
        snapshot_points='0.2,0.4,0.6,0.8,1.0',
        snapshot_reduce='quantile',
        snapshot_quantile=0.2,
        snapshot_agreement_lambda=0.0,
        snapshot_dynamic_calibration=False,
        snapshot_calib_beta=0.2,
        snapshot_calib_rho_max=0.1,
        snapshot_calib_tau=0.5,
        snapshot_calib_norm_scope='task',
        snapshot_calib_use_confidence=False,
        snapshot_calib_use_variability=False,
        snapshot_calib_use_forgetting=False,
        snapshot_calib_conf_gate='linear',
        snapshot_calib_conf_gamma=1.0,
        snapshot_calib_conf_eps=1e-6,
        snapshot_calib_conf_log_lambda=0.2,
        snapshot_calib_lowconf_q=0.20,
        snapshot_calib_lowvar_tail_q=0.05,
        snapshot_calib_lowvar_eta=0.02,
        snapshot_calib_highconf_protect_q=0.80,
        snapshot_verbose=False,
        div_lambda=0.1,
        div_candidate_ratio=3,
        div_feature_source='current_model',
        div_feature_layer='penultimate',
        div_verbose=False,
        filter_sim_threshold=0.85,
        filter_feature_source='current_model',
        filter_feature_layer='penultimate',
        filter_reject_dominated=False,
        filter_verbose=False,
        ref_train_lr=0.1,
        ref_train_epoch=1,
        use_cuda=0,
        ref_sample_per_task=0
    )


def _make_agent(local_path):
    return selection_agent.RhoSelectionAgent(
        local_path=local_path,
        transforms=None,
        init_size=0,
        selection_steps=1,
        cur_train_lr=0.1,
        cur_train_steps=1,
        use_cuda=False,
        eval_mode='none',
        early_stop=-1,
        eval_steps=1,
        model_params=MODEL_PARAMS,
        ref_train_params={
            'lr': 0.1,
            'epochs': 1,
            'batch_size': 2,
            'eval_batch_size': 2,
            'use_cuda': False,
            'early_stop': -1,
            'log_steps': 100,
            'opt_type': 'sgd',
            'loss_params': {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            },
            'ref_sample_per_task': 0
        },
        seed=0,
        class_balance=False,
        only_new_data=True,
        loss_params={
            'ce_factor': 1.0,
            'mse_factor': 0.0
        }
    )


def _cpu_state_dict(model):
    return {
        name: tensor.clone().detach().cpu()
        for name, tensor in model.state_dict().items()
    }


def TEST_ARGPARSE_HAS_SNAPSHOT_ARGS():
    output = subprocess.check_output(
        [sys.executable, 'offline_continual_learning.py', '--help'],
        stderr=subprocess.STDOUT
    ).decode('utf-8', errors='ignore')
    for key in [
            '--rel_ref_mode',
            '--snapshot_points',
            '--snapshot_reduce',
            '--snapshot_quantile',
            '--snapshot_agreement_lambda',
            '--snapshot_dynamic_calibration',
            '--snapshot_calib_beta',
            '--snapshot_calib_rho_max',
            '--snapshot_calib_tau',
            '--snapshot_calib_norm_scope',
            '--snapshot_calib_use_confidence',
            '--snapshot_calib_use_variability',
            '--snapshot_calib_use_forgetting',
            '--snapshot_calib_conf_gate',
            '--snapshot_calib_conf_gamma',
            '--snapshot_calib_conf_eps',
            '--snapshot_calib_conf_log_lambda',
            '--snapshot_calib_lowconf_q',
            '--snapshot_calib_lowvar_tail_q',
            '--snapshot_calib_lowvar_eta',
            '--snapshot_calib_highconf_protect_q']:
        assert key in output
    print('TEST_ARGPARSE_HAS_SNAPSHOT_ARGS PASS')


def TEST_SNAPSHOT_REDUCE():
    losses = [3.0, 2.0, 1.0, 0.5, 0.4]
    assert abs(funcs.reduce_snapshot_losses(losses, 'min', 0.2) - 0.4) < 1e-6
    assert abs(funcs.reduce_snapshot_losses(losses, 'median', 0.2) - 1.0) < 1e-6
    assert abs(funcs.reduce_snapshot_losses(losses, 'quantile', 0.2) - 0.48) < 1e-6
    print('TEST_SNAPSHOT_REDUCE PASS')


def TEST_AGREEMENT_PENALTY():
    losses = [3.0, 2.0, 1.0, 0.5, 0.4]
    base = funcs.reduce_snapshot_losses(losses, 'quantile', 0.2)
    proxy = funcs.compute_snapshot_proxy_loss(
        losses=losses,
        snapshot_reduce='quantile',
        snapshot_quantile=0.2,
        agreement=0.6,
        agreement_lambda=1.0
    )
    assert abs(proxy - (base + 0.4)) < 1e-6
    print('TEST_AGREEMENT_PENALTY PASS')


def TEST_DEFAULT_TRAINED_REF_UNCHANGED():
    params = offline_continual_learning.make_selection_params(opts=_make_opts())
    assert params['rel_ref_mode'] == 'trained_ref'
    assert params['snapshot_reduce'] == 'quantile'
    assert params['snapshot_dynamic_calibration'] is False
    assert params['snapshot_calib_use_confidence'] is False
    assert params['snapshot_calib_use_variability'] is False
    assert params['snapshot_calib_use_forgetting'] is False
    assert params['snapshot_calib_conf_gate'] == 'linear'
    assert params['snapshot_calib_conf_gamma'] == 1.0
    assert params['snapshot_calib_conf_eps'] == 1e-6
    assert params['snapshot_calib_conf_log_lambda'] == 0.2
    assert params['snapshot_calib_lowconf_q'] == 0.20
    assert params['snapshot_calib_lowvar_tail_q'] == 0.05
    assert params['snapshot_calib_lowvar_eta'] == 0.02
    assert params['snapshot_calib_highconf_protect_q'] == 0.80
    print('TEST_DEFAULT_TRAINED_REF_UNCHANGED PASS')


def TEST_EMPTY_SNAPSHOT_REJECTED():
    temp_dir = tempfile.mkdtemp(prefix='snapshot_diag_')
    try:
        agent = _make_agent(temp_dir)
        x = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        y = np.asarray([0, 1], dtype=np.int64)
        try:
            agent.compute_ref_loss_dic_from_snapshots(
                x=x,
                y=y,
                snapshot_state_dicts=[],
                id_list=[3, 4]
            )
        except ValueError:
            print('TEST_EMPTY_SNAPSHOT_REJECTED PASS')
            return
        raise AssertionError('expected ValueError for empty snapshot_state_dicts')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def TEST_PROXY_REF_KEYS_ALIGN():
    temp_dir = tempfile.mkdtemp(prefix='snapshot_diag_')
    try:
        torch.manual_seed(0)
        agent = _make_agent(temp_dir)
        x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y = np.asarray([0, 1, 0], dtype=np.int64)
        id_list = [10, 11, 12]
        model = utils.build_model(MODEL_PARAMS)
        final_loss_dic = agent.compute_ref_loss_dic_from_model(
            x=x,
            y=y,
            ref_proxy_model=model,
            id_list=id_list
        )
        snapshot_loss_dic = agent.compute_ref_loss_dic_from_snapshots(
            x=x,
            y=y,
            snapshot_state_dicts=[_cpu_state_dict(model)],
            id_list=id_list,
            snapshot_reduce='quantile',
            snapshot_quantile=0.2,
            snapshot_agreement_lambda=0.0
        )
        assert set(final_loss_dic.keys()) == set(id_list)
        assert set(snapshot_loss_dic.keys()) == set(id_list)
        print('TEST_PROXY_REF_KEYS_ALIGN PASS')
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def TEST_DYNAMIC_CALIBRATION():
    losses = [3.0, 2.0, 1.0, 0.5, 0.4]
    q_alpha = float(np.quantile(losses, 0.1))
    q_beta = float(np.quantile(losses, 0.2))
    low_risk_ref, low_risk_details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=losses,
        p_trues=[0.92, 0.91, 0.90, 0.93, 0.92],
        corrects=[1, 1, 1, 1, 1],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=0.5,
        var_norm=0.5,
        return_details=True
    )
    high_risk_ref, high_risk_details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=losses,
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=0.5,
        var_norm=0.5,
        return_details=True
    )
    ref_zero = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=losses,
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.0,
        tau=0.5,
        var_norm=0.5
    )
    assert abs(ref_zero - q_alpha) < 1e-6
    assert high_risk_details['risk'] > low_risk_details['risk']
    assert high_risk_ref > low_risk_ref
    assert q_alpha <= low_risk_ref <= q_beta
    assert q_alpha <= high_risk_ref <= q_beta
    try:
        funcs.compute_dynamic_calibrated_snapshot_ref_loss(
            snapshot_losses=losses,
            p_trues=[0.5] * 5,
            corrects=[1] * 5,
            alpha=0.2,
            beta=0.2
        )
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for beta <= alpha')
    ref_loss_dic, diagnostics = funcs.compute_dynamic_calibrated_snapshot_ref_loss_dic(
        id2losses={10: losses, 11: losses},
        id2p_trues={10: [0.9] * 5, 11: [0.2] * 5},
        id2corrects={10: [1] * 5, 11: [1, 0, 1, 0, 0]},
        alpha=0.1,
        beta=0.2,
        rho_max=0.0,
        tau=0.5,
        norm_scope='task'
    )
    assert abs(ref_loss_dic[10] - q_alpha) < 1e-6
    assert abs(ref_loss_dic[11] - q_alpha) < 1e-6
    assert diagnostics['var_norm'] == [0.5, 0.5]
    print('TEST_DYNAMIC_CALIBRATION PASS')



def TEST_DYNAMIC_CALIBRATION_ABLATION_SWITCHES():
    conf_risk = 0.8
    var_risk = 0.4
    forg_risk = 0.5
    tau = 0.5
    full = funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau)
    assert abs(full - (conf_risk * var_risk + tau * conf_risk * forg_risk)) < 1e-12
    assert abs(funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_confidence=True) - conf_risk) < 1e-12
    assert abs(funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_variability=True) - var_risk) < 1e-12
    assert abs(funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_forgetting=True) - forg_risk) < 1e-12
    assert abs(funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_confidence=True, use_variability=True) - conf_risk * var_risk) < 1e-12
    assert abs(funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_confidence=True, use_forgetting=True) - conf_risk * forg_risk) < 1e-12
    assert abs(funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_variability=True, use_forgetting=True) - var_risk * forg_risk) < 1e-12
    full_switch = funcs.compute_snapshot_dynamic_calibration_risk(
        conf_risk, var_risk, forg_risk, tau=tau,
        use_confidence=True, use_variability=True, use_forgetting=True)
    assert abs(full_switch - full) < 1e-12
    _, details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=[3.0, 2.0, 1.0, 0.5, 0.4],
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=tau,
        var_norm=0.5,
        use_confidence=True,
        return_details=True
    )
    assert details['ablation_mode'] is True
    assert details['use_confidence'] is True
    assert details['use_variability'] is False
    assert details['use_forgetting'] is False
    print('TEST_DYNAMIC_CALIBRATION_ABLATION_SWITCHES PASS')


def TEST_DYNAMIC_CALIBRATION_FOCAL_LOG_CONFIDENCE_GATE():
    conf = 0.25
    linear = funcs.compute_snapshot_confidence_risk(conf, conf_gate='linear')
    focal = funcs.compute_snapshot_confidence_risk(
        conf, conf_gate='focal_log', conf_gamma=1.0, conf_eps=1e-6)
    expected = min(max((1.0 - conf) * (-np.log(conf)) / np.log(2.0), 0.0), 1.0)
    assert abs(linear - 0.75) < 1e-12
    assert abs(focal - expected) < 1e-12
    _, linear_details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=[3.0, 2.0, 1.0, 0.5, 0.4],
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=0.5,
        var_norm=0.5,
        use_confidence=True,
        return_details=True
    )
    _, focal_details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=[3.0, 2.0, 1.0, 0.5, 0.4],
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=0.5,
        var_norm=0.5,
        use_confidence=True,
        conf_gate='focal_log',
        conf_gamma=1.0,
        conf_eps=1e-6,
        return_details=True
    )
    _, full_details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=[3.0, 2.0, 1.0, 0.5, 0.4],
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=0.5,
        var_norm=0.5,
        conf_gate='focal_log',
        conf_gamma=1.0,
        conf_eps=1e-6,
        return_details=True
    )
    assert linear_details['conf_gate'] == 'linear'
    assert focal_details['conf_gate'] == 'focal_log'
    assert abs(linear_details['conf_risk'] - (1.0 - linear_details['conf'])) < 1e-12
    assert focal_details['conf_risk'] >= linear_details['conf_risk']
    assert full_details['ablation_mode'] is False
    assert abs(full_details['conf_risk'] - (1.0 - full_details['conf'])) < 1e-12
    print('TEST_DYNAMIC_CALIBRATION_FOCAL_LOG_CONFIDENCE_GATE PASS')


def TEST_DYNAMIC_CALIBRATION_LINEAR_LOG_RESIDUAL_CONFIDENCE_GATE():
    conf = 0.25
    eps = 1e-6
    linear = funcs.compute_snapshot_confidence_risk(conf, conf_gate='linear')
    focal = funcs.compute_snapshot_confidence_risk(
        conf, conf_gate='focal_log', conf_gamma=1.0, conf_eps=eps)
    llr_zero = funcs.compute_snapshot_confidence_risk(
        conf, conf_gate='linear_log_residual', conf_eps=eps, conf_log_lambda=0.0)
    llr = funcs.compute_snapshot_confidence_risk(
        conf, conf_gate='linear_log_residual', conf_eps=eps, conf_log_lambda=0.2)
    expected_llr = np.clip(
        (1.0 - conf) *
        (1.0 + 0.2 * (-np.log(max(conf, eps)) / -np.log(eps))),
        0.0,
        1.0
    )
    assert abs(llr_zero - linear) < 1e-12
    assert abs(llr - expected_llr) < 1e-12
    assert linear <= llr < focal

    for test_conf in [0.0, 0.05, 0.25, 0.5, 0.9, 1.0]:
        risks = [
            funcs.compute_snapshot_confidence_risk(test_conf, conf_gate='linear'),
            funcs.compute_snapshot_confidence_risk(
                test_conf, conf_gate='focal_log', conf_gamma=1.0, conf_eps=eps),
            funcs.compute_snapshot_confidence_risk(
                test_conf, conf_gate='linear_log_residual',
                conf_eps=eps, conf_log_lambda=0.0),
            funcs.compute_snapshot_confidence_risk(
                test_conf, conf_gate='linear_log_residual',
                conf_eps=eps, conf_log_lambda=0.2)
        ]
        assert all(0.0 <= risk <= 1.0 for risk in risks)
        assert abs(risks[2] - risks[0]) < 1e-12
        assert risks[3] >= risks[0]

    _, full_details = funcs.compute_dynamic_calibrated_snapshot_ref_loss(
        snapshot_losses=[3.0, 2.0, 1.0, 0.5, 0.4],
        p_trues=[0.20, 0.24, 0.18, 0.22, 0.19],
        corrects=[1, 0, 1, 0, 0],
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        tau=0.5,
        var_norm=0.5,
        conf_gate='linear_log_residual',
        conf_eps=eps,
        conf_log_lambda=0.2,
        return_details=True
    )
    expected_full = (
        (1.0 - full_details['conf']) * (1.0 - full_details['var_norm']) +
        0.5 * full_details['forg'] * (1.0 - full_details['conf'])
    )
    assert full_details['ablation_mode'] is False
    assert abs(full_details['conf_risk'] - (1.0 - full_details['conf'])) < 1e-12
    assert abs(full_details['risk'] - expected_full) < 1e-12
    print('TEST_DYNAMIC_CALIBRATION_LINEAR_LOG_RESIDUAL_CONFIDENCE_GATE PASS')


def TEST_DYNAMIC_CALIBRATION_LLR_LOWVAR_TAIL_GATE():
    conf_values = np.asarray([0.10, 0.20, 0.90, 0.80], dtype=np.float64)
    var_norm_values = np.asarray([0.00, 0.80, 0.10, 0.90], dtype=np.float64)
    context = funcs.compute_snapshot_lowvar_tail_context(
        conf_values=conf_values,
        var_norm_values=var_norm_values,
        lowconf_q=0.50,
        lowvar_tail_q=0.40
    )
    gate_llr = np.asarray([
        funcs.compute_snapshot_confidence_risk(
            conf, conf_gate='linear_log_residual',
            conf_eps=1e-6, conf_log_lambda=0.2)
        for conf in conf_values
    ])
    gate_eta_zero = np.asarray([
        funcs.compute_snapshot_confidence_risk(
            conf,
            conf_gate='linear_log_residual_lowvar_tail',
            conf_eps=1e-6,
            conf_log_lambda=0.2,
            lowconf_mask=context['lowconf_mask'][idx],
            lowvar_tail_score=context['lowvar_tail_score'][idx],
            lowvar_eta=0.0
        )
        for idx, conf in enumerate(conf_values)
    ])
    gate_final = np.asarray([
        funcs.compute_snapshot_confidence_risk(
            conf,
            conf_gate='linear_log_residual_lowvar_tail',
            conf_eps=1e-6,
            conf_log_lambda=0.2,
            lowconf_mask=context['lowconf_mask'][idx],
            lowvar_tail_score=context['lowvar_tail_score'][idx],
            lowvar_eta=0.02
        )
        for idx, conf in enumerate(conf_values)
    ])

    assert np.array_equal(gate_eta_zero, gate_llr)
    assert context['lowconf_mask'].tolist() == [True, True, False, False]
    assert context['lowvar_tail_score'][2] > 0.0
    assert gate_final[2] == gate_llr[2]
    assert context['lowvar_tail_score'][1] == 0.0
    assert gate_final[1] == gate_llr[1]
    assert gate_final[0] > gate_llr[0]
    assert gate_final[0] - gate_llr[0] <= 0.02 * (1.0 - gate_llr[0])
    assert np.all(gate_final >= 0.0)
    assert np.all(gate_final <= 1.0)

    single = funcs.compute_snapshot_lowvar_tail_context([0.1], [0.0])
    equal = funcs.compute_snapshot_lowvar_tail_context(
        [0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    disabled = funcs.compute_snapshot_lowvar_tail_context(
        [0.1, 0.2], [0.0, 1.0], lowvar_tail_q=0.0)
    assert single['lowvar_tail_score'].tolist() == [0.0]
    assert np.all(equal['lowvar_tail_score'] == 0.0)
    assert np.all(disabled['lowvar_tail_score'] == 0.0)

    _, diagnostics = funcs.compute_dynamic_calibrated_snapshot_ref_loss_dic(
        id2losses={idx: [3.0, 2.0, 1.0, 0.5, 0.4] for idx in range(4)},
        id2p_trues={
            0: [0.098, 0.099, 0.100, 0.101, 0.102],
            1: [0.00, 0.10, 0.20, 0.30, 0.40],
            2: [0.88, 0.89, 0.90, 0.91, 0.92],
            3: [0.50, 0.625, 0.75, 0.875, 1.00]
        },
        id2corrects={idx: [1, 0, 1, 0, 0] for idx in range(4)},
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        use_confidence=True,
        conf_gate='linear_log_residual_lowvar_tail',
        conf_log_lambda=0.2,
        lowconf_q=0.50,
        lowvar_tail_q=0.40,
        lowvar_eta=0.02
    )
    assert diagnostics['use_confidence'] is True
    assert diagnostics['use_variability'] is False
    assert diagnostics['lowconf_count'] == 2
    assert diagnostics['lowvar_tail_count'] == 2
    assert diagnostics['joint_lowconf_lowvar_count'] == 1
    assert diagnostics['gate_final'][0] > diagnostics['gate_llr'][0]
    assert diagnostics['gate_final'][1] == diagnostics['gate_llr'][1]
    assert diagnostics['gate_final'][2] == diagnostics['gate_llr'][2]

    implementation = (
        inspect.getsource(funcs.compute_snapshot_lowvar_tail_context) +
        inspect.getsource(funcs.compute_snapshot_confidence_risk)
    )
    assert '1.0 - var_norm' not in implementation
    assert '(1 - var_norm' not in implementation
    print('TEST_DYNAMIC_CALIBRATION_LLR_LOWVAR_TAIL_GATE PASS')


def TEST_DYNAMIC_CALIBRATION_LLR_PROTECTED_LOWVAR_TAIL_GATE():
    conf_values = np.asarray([0.10, 0.20, 0.90, 0.80], dtype=np.float64)
    var_norm_values = np.asarray([0.00, 0.80, 0.10, 0.90], dtype=np.float64)
    context = funcs.compute_snapshot_protected_lowvar_tail_context(
        conf_values=conf_values,
        var_norm_values=var_norm_values,
        highconf_protect_q=0.75,
        lowvar_tail_q=0.40
    )
    gate_llr = np.asarray([
        funcs.compute_snapshot_confidence_risk(
            conf, conf_gate='linear_log_residual',
            conf_eps=1e-6, conf_log_lambda=0.2)
        for conf in conf_values
    ])
    gate_eta_zero = np.asarray([
        funcs.compute_snapshot_confidence_risk(
            conf,
            conf_gate='linear_log_residual_protected_lowvar_tail',
            conf_eps=1e-6,
            conf_log_lambda=0.2,
            not_highconf_mask=context['not_highconf_mask'][idx],
            lowvar_tail_score=context['lowvar_tail_score'][idx],
            lowvar_eta=0.0
        )
        for idx, conf in enumerate(conf_values)
    ])
    gate_final = np.asarray([
        funcs.compute_snapshot_confidence_risk(
            conf,
            conf_gate='linear_log_residual_protected_lowvar_tail',
            conf_eps=1e-6,
            conf_log_lambda=0.2,
            not_highconf_mask=context['not_highconf_mask'][idx],
            lowvar_tail_score=context['lowvar_tail_score'][idx],
            lowvar_eta=0.05
        )
        for idx, conf in enumerate(conf_values)
    ])

    assert np.array_equal(gate_eta_zero, gate_llr)
    assert context['not_highconf_mask'].tolist() == [True, True, False, True]
    assert context['lowvar_tail_score'][2] > 0.0
    assert gate_final[2] == gate_llr[2]
    assert context['lowvar_tail_score'][0] > 0.0
    assert gate_final[0] > gate_llr[0]
    assert gate_final[0] - gate_llr[0] <= 0.05 * (1.0 - gate_llr[0])
    assert context['lowvar_tail_score'][1] == 0.0
    assert gate_final[1] == gate_llr[1]
    assert np.all(gate_final >= 0.0)
    assert np.all(gate_final <= 1.0)

    protect_all = funcs.compute_snapshot_protected_lowvar_tail_context(
        conf_values, var_norm_values, highconf_protect_q=0.0,
        lowvar_tail_q=0.40)
    protect_top = funcs.compute_snapshot_protected_lowvar_tail_context(
        conf_values, var_norm_values, highconf_protect_q=1.0,
        lowvar_tail_q=0.40)
    single = funcs.compute_snapshot_protected_lowvar_tail_context(
        [0.1], [0.0])
    equal_conf = funcs.compute_snapshot_protected_lowvar_tail_context(
        [0.5, 0.5, 0.5], [0.0, 0.5, 1.0])
    equal_var = funcs.compute_snapshot_protected_lowvar_tail_context(
        [0.1, 0.2, 0.3], [0.5, 0.5, 0.5])
    disabled = funcs.compute_snapshot_protected_lowvar_tail_context(
        [0.1, 0.2], [0.0, 1.0], lowvar_tail_q=0.0)
    assert not np.any(protect_all['not_highconf_mask'])
    assert np.sum(protect_top['not_highconf_mask']) == 3
    assert single['lowvar_tail_score'].tolist() == [0.0]
    assert not np.any(equal_conf['not_highconf_mask'])
    assert np.all(equal_var['lowvar_tail_score'] == 0.0)
    assert np.all(disabled['lowvar_tail_score'] == 0.0)

    _, diagnostics = funcs.compute_dynamic_calibrated_snapshot_ref_loss_dic(
        id2losses={idx: [3.0, 2.0, 1.0, 0.5, 0.4] for idx in range(4)},
        id2p_trues={
            0: [0.098, 0.099, 0.100, 0.101, 0.102],
            1: [0.00, 0.10, 0.20, 0.30, 0.40],
            2: [0.88, 0.89, 0.90, 0.91, 0.92],
            3: [0.50, 0.625, 0.75, 0.875, 1.00]
        },
        id2corrects={idx: [1, 0, 1, 0, 0] for idx in range(4)},
        alpha=0.1,
        beta=0.2,
        rho_max=0.1,
        use_confidence=True,
        conf_gate='linear_log_residual_protected_lowvar_tail',
        conf_log_lambda=0.2,
        lowvar_tail_q=0.40,
        lowvar_eta=0.05,
        highconf_protect_q=0.75
    )
    assert diagnostics['use_confidence'] is True
    assert diagnostics['use_variability'] is False
    assert diagnostics['highconf_protected_count'] == 1
    assert diagnostics['not_highconf_count'] == 3
    assert diagnostics['lowvar_tail_count'] == 2
    assert diagnostics['active_protected_lowvar_count'] == 1
    assert diagnostics['gate_final'][0] > diagnostics['gate_llr'][0]
    assert diagnostics['gate_final'][1] == diagnostics['gate_llr'][1]
    assert diagnostics['gate_final'][2] == diagnostics['gate_llr'][2]

    implementation = (
        inspect.getsource(funcs.compute_snapshot_protected_lowvar_tail_context) +
        inspect.getsource(funcs.compute_snapshot_confidence_risk)
    )
    assert '1.0 - var_norm' not in implementation
    assert '(1 - var_norm' not in implementation
    print('TEST_DYNAMIC_CALIBRATION_LLR_PROTECTED_LOWVAR_TAIL_GATE PASS')


def main():
    TEST_ARGPARSE_HAS_SNAPSHOT_ARGS()
    TEST_SNAPSHOT_REDUCE()
    TEST_AGREEMENT_PENALTY()
    TEST_DEFAULT_TRAINED_REF_UNCHANGED()
    TEST_EMPTY_SNAPSHOT_REJECTED()
    TEST_PROXY_REF_KEYS_ALIGN()
    TEST_DYNAMIC_CALIBRATION()
    TEST_DYNAMIC_CALIBRATION_ABLATION_SWITCHES()
    TEST_DYNAMIC_CALIBRATION_FOCAL_LOG_CONFIDENCE_GATE()
    TEST_DYNAMIC_CALIBRATION_LINEAR_LOG_RESIDUAL_CONFIDENCE_GATE()
    TEST_DYNAMIC_CALIBRATION_LLR_LOWVAR_TAIL_GATE()
    TEST_DYNAMIC_CALIBRATION_LLR_PROTECTED_LOWVAR_TAIL_GATE()
    print('PASS synthetic_snapshot_holdout_diagnostic')


if __name__ == '__main__':
    main()
