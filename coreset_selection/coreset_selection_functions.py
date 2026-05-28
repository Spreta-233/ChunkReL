# -*-coding:utf8-*-

import os
import torch
from torch.utils.data import DataLoader
import numpy as np
import random
import pickle
import copy

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
        'gss_anchor_replace_historical_fallback_count': 0,
        'gss_anchor_replace_grad_layer_used': ''
    }


def gss_anchor_replace_uses_anchor(selection_strategy, is_current_task):
    return selection_strategy == 'rel_gss_anchor_replace' and bool(is_current_task)


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


def select_anchor_replace_ranked_items(sorted_loss_diffs, gradients_by_id, window_size=8, anchor_size=4,
                                       sim_threshold=0.90, stats=None):
    window_size = max(int(window_size), 1)
    anchor_size = max(int(anchor_size), 1)
    anchor_size = min(anchor_size, len(sorted_loss_diffs))
    window_size = max(window_size, anchor_size)
    window_items = list(sorted_loss_diffs[:min(window_size, len(sorted_loss_diffs))])
    selected_items = list(window_items[:anchor_size])
    tail_items = list(window_items[anchor_size:])
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_chunks_total', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_current_task_chunks', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_tail_candidates_total', len(tail_items))
    for candidate_item in tail_items:
        cand_id, cand_rel = candidate_item
        cand_grad = gradients_by_id[int(cand_id)]
        sim_values = []
        for selected_item in selected_items:
            selected_id = int(selected_item[0])
            selected_grad = gradients_by_id[selected_id]
            sim_value = float(torch.dot(_normalize_grad_vector(cand_grad), _normalize_grad_vector(selected_grad)))
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
        selected_items[replace_pos] = candidate_item
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_replaced_total', 1)
        _add_gss_anchor_replace_stat(
            stats, 'gss_anchor_replace_rel_gap_sum', float(replace_target[1]) - float(cand_rel))
        _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_rel_gap_count', 1)
    _add_gss_anchor_replace_stat(stats, 'gss_anchor_replace_final_kept_total', len(selected_items))
    return selected_items


def select_by_loss_diff_with_anchor_replace(ref_loss_dic, rand_data, model, incremental_size, transforms, on_cuda,
                                            loss_params, class_sizes=None, window_size=8, anchor_size=4,
                                            sim_threshold=0.90, grad_layer='classifier', stats=None):
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
        stats=stats
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
