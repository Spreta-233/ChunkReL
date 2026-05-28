# -*-coding:utf8-*-

import torch
from torch.utils.data import DataLoader
import torchvision
import pickle
import os
import random
import copy

import utils
from dataset import single_task_dataset
from coreset_selection import coreset_selection_functions
from coreset_selection import train_methods_for_selection
from functions import train_methods


class RhoSelectionAgent(object):
    def __init__(self, local_path, transforms, init_size, selection_steps, cur_train_lr, cur_train_steps, use_cuda,
                 eval_mode, early_stop, eval_steps, model_params, ref_train_params, seed, ref_model=None,
                 class_balance=True, only_new_data=True, loss_params=None, save_checkpoint=False,
                 selection_strategy='rel', selection_chunk_size=0, div_lambda=0.1, div_candidate_ratio=3,
                 div_feature_source='current_model', div_feature_layer='penultimate', div_verbose=False,
                 filter_sim_threshold=0.85, ccrf_sim_threshold=0.85, filter_feature_source='current_model',
                 filter_feature_layer='penultimate', filter_reject_dominated=False, filter_verbose=False,
                 gss_grad_sim_threshold=0.95, gss_grad_layer='classifier'):
        # all related setting
        self.local_path = local_path
        if not os.path.exists(self.local_path):
            os.makedirs(self.local_path)
        self.transforms = transforms
        self.init_size = init_size
        self.selection_steps = selection_steps
        self.cur_train_lr = cur_train_lr
        self.cur_train_steps = cur_train_steps
        self.eval_mode = eval_mode
        self.early_stop = early_stop
        self.eval_steps = eval_steps
        self.class_balance = class_balance
        self.only_new_data = only_new_data
        self.loss_params = loss_params
        self.selection_strategy = selection_strategy
        self.selection_chunk_size = selection_chunk_size
        self.div_lambda = div_lambda
        self.div_candidate_ratio = div_candidate_ratio
        self.div_feature_source = div_feature_source
        self.div_feature_layer = div_feature_layer
        self.div_verbose = div_verbose
        self.filter_sim_threshold = filter_sim_threshold
        self.ccrf_sim_threshold = ccrf_sim_threshold
        self.gss_grad_sim_threshold = gss_grad_sim_threshold
        self.gss_grad_layer = gss_grad_layer
        self.filter_feature_source = filter_feature_source
        self.filter_feature_layer = filter_feature_layer
        self.filter_reject_dominated = filter_reject_dominated
        self.filter_verbose = filter_verbose
        self.ref_model = ref_model
        self.ref_train_params = ref_train_params
        self.model_params = model_params
        self.seed = seed
        self.save_checkpoint = save_checkpoint
        # make train_params
        if loss_params is None:
            loss_params = {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            }
        self.train_params = {
            'lr': self.cur_train_lr,
            'steps': self.cur_train_steps,
            'batch_size': 32,
            'eval_batch_size': 20,
            'use_cuda': use_cuda,
            'early_stop': self.early_stop,
            'log_steps': 100,
            'opt_type': 'sgd',
            'loss_params': loss_params
        }
        self.cur_train_file = os.path.join(self.local_path, 'cur_train.pkl')
        self.to_pil = torchvision.transforms.ToPILImage()

    def make_data_loader(self, x, y, fname, batch_size, id_list=None, id2logit=None, extra_data=None):
        """
        make train-dataloader from numpy array inputs
        :param x: input numpy array
        :param y: target numpy array
        :param fname:
        :param batch_size:
        :param id_list:
        :param id2logit:
        :param extra_data:
        :return:
        """
        data_size = x.shape[0]
        data_file = os.path.join(self.local_path, fname)
        with open(data_file, 'wb') as fw:
            for i in range(data_size):
                if self.transforms is not None:
                    sp = self.to_pil(torch.tensor(x[i], dtype=torch.float32).clone().detach())
                else:
                    sp = torch.tensor(x[i], dtype=torch.float32).clone().detach()
                if id_list is not None:
                    data = [id_list[i], sp, int(y[i])]
                    if id2logit is not None:
                        data.append(id2logit[id_list[i]])
                else:
                    data = [i, sp, int(y[i])]
                    if id2logit is not None:
                        data.append(id2logit[i])
                pickle.dump(data, fw)
            if extra_data is not None:
                for di in extra_data:
                    if len(di) == 4 and id2logit is None:
                        pickle.dump(di[:3], fw)
                    else:
                        pickle.dump(di, fw)
        dataset = single_task_dataset.PILDataset(
            local_path=self.local_path,
            data_path=data_file,
            transforms=self.transforms
        )
        dataset.set_produce_id(produce_id=True)
        dataset.shuffle_dataset()
        data_loader = DataLoader(dataset, batch_size=batch_size, drop_last=False)
        return data_loader, data_file

    def train_ref_model(self, x, y, verbose=True, id2logit=None, ideal_logit=False, extra_data=None, log_file=None):
        print('=== train holdout model ===')
        train_loader, data_file = self.make_data_loader(
            x=x,
            y=y,
            fname='ref_train.pkl',
            batch_size=self.ref_train_params['batch_size'],
            id2logit=id2logit,
            extra_data=extra_data
        )
        init_model = utils.build_model(model_params=self.model_params)
        if ideal_logit and 'loss_params' in self.ref_train_params:
            temp_train_params = copy.deepcopy(self.ref_train_params)
            temp_train_params['loss_params'] = {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            }
        else:
            temp_train_params = self.ref_train_params
        trained_model = train_methods.train_model(
            local_path=self.local_path,
            model=init_model,
            train_loader=train_loader,
            eval_loader=None,
            epochs=self.ref_train_params['epochs'],
            train_params=temp_train_params,
            verbose=verbose,
            save_ckpt=False,
            load_best=False,
            weight_decay=0,
            log_file=log_file
        )
        os.remove(data_file)
        self.ref_model = trained_model

    def incremental_selection(self, x, y, select_size, id_list=None, loss_dic=None, loss_dic_dump_file=None,
                              verbose=True, class_pool=None, id2logit=None, ideal_logit=False, extra_data=None,
                              selected_task_id=None, current_training_task_id=None):
        if select_size >= x.shape[0]:
            print('Warning: select size greater than data size', select_size, x.shape[0])
        # the id of each sample is assigned according to order.
        if loss_dic is None:
            # train reference model
            if self.ref_model is None:
                self.train_ref_model(x=x, y=y, id2logit=id2logit)
            # compute loss dict
            temp_loader, temp_file = self.make_data_loader(
                x=x,
                y=y,
                fname='temp_data.pkl',
                batch_size=20,
                id_list=id_list,
                id2logit=id2logit
            )
            if ideal_logit:
                ref_loss_params = {
                    'ce_factor': 1.0,
                    'mse_factor': 0.0
                }
            else:
                ref_loss_params = self.train_params['loss_params']
            ref_loss_dic = utils.compute_loss_dic(
                ref_model=self.ref_model,
                data_loader=temp_loader,
                aug_iters=1,
                use_cuda=self.train_params['use_cuda'],
                loss_params=ref_loss_params
            )
            if loss_dic_dump_file is not None:
                with open(loss_dic_dump_file, 'wb') as fw:
                    pickle.dump(ref_loss_dic, fw)
            os.remove(temp_file)
        else:
            ref_loss_dic = loss_dic
        # init model and selection
        all_selected_ids = set()
        base_incremental_size = max(int(select_size / self.selection_steps), 1)
        if selected_task_id is not None and current_training_task_id is not None:
            ccrf_is_current_task = int(selected_task_id) == int(current_training_task_id)
        else:
            ccrf_is_current_task = True
        init_model = utils.build_model(model_params=self.model_params)
        all_class_ids = get_class_dic(y=y)
        class_ids = {}
        if class_pool is None:
            for i in range(self.model_params['num_class']):
                class_ids[i] = set()
        else:
            for i in class_pool:
                class_ids[i] = set()
        cur_train_dataset = single_task_dataset.RandomDataset(
            seed=self.seed,
            data_path=self.cur_train_file,
            transforms=self.transforms,  # the transforms can be altered here
            extra_data=extra_data
        )
        train_loader = DataLoader(cur_train_dataset, batch_size=self.train_params['batch_size'], drop_last=False)
        if self.eval_mode in ['acc', 'avg_loss', 'loss_var']:
            full_train_loader, full_data_file = self.make_data_loader(
                x=x,
                y=y,
                batch_size=self.train_params['batch_size'],
                fname='full_data.pkl',
                id_list=id_list,
                id2logit=id2logit
            )
        else:
            full_train_loader = None
            full_data_file = ''
        if id_list is None:
            full_ids = list(range(x.shape[0]))
        else:
            full_ids = id_list
        max_rejected_total = max(len(full_ids) - select_size, 0)
        div_selected_features_bank = []
        div_selected_ids_bank = []
        div_stateful_diag = {
            'calls': 0,
            'selection_sizes': [],
            'max_similarity_nonzero_after_first': False,
            'div_penalty_active': False
        }
        rejected_ids = set()
        rejection_filter_strategies = ['rel_filter', 'rel_ccrf']
        ccrf_stats = {
            'ccrf_enabled': self.selection_strategy == 'rel_ccrf',
            'ccrf_sim_threshold': self.ccrf_sim_threshold,
            'ccrf_chunk_size': self.selection_chunk_size,
            'ccrf_chunks_total': 0,
            'ccrf_chunks_with_rejection': 0,
            'ccrf_rejected_total': 0,
            'ccrf_kept_total': 0,
            'ccrf_same_class_pair_checked_count': 0,
            'ccrf_same_class_pair_rejected_count': 0,
            'ccrf_cross_class_pair_ignored_count': 0,
            'ccrf_current_task_chunks': 0,
            'ccrf_historical_task_chunks': 0,
            'ccrf_historical_fallback_enabled': self.selection_strategy == 'rel_ccrf',
            'ccrf_historical_fallback_count': 0,
            'ccrf_effective_chunk_size_current': self.selection_chunk_size if self.selection_chunk_size > 0
            else base_incremental_size,
            'ccrf_effective_chunk_size_historical': 1
        }
        gss_temp_stats = {
            'gss_temp_enabled': self.selection_strategy == 'rel_gss_temp',
            'gss_temp_grad_sim_threshold': self.gss_grad_sim_threshold,
            'gss_temp_chunk_size': self.selection_chunk_size,
            'gss_temp_grad_layer': self.gss_grad_layer,
            'gss_temp_chunks_total': 0,
            'gss_temp_chunks_with_deferral': 0,
            'gss_temp_deferred_total': 0,
            'gss_temp_permanent_rejected_total': 0,
            'gss_temp_kept_total': 0,
            'gss_temp_grad_pair_checked_count': 0,
            'gss_temp_grad_pair_deferred_count': 0,
            'gss_temp_grad_layer_used': '',
            'gss_temp_grad_cos_sum': 0.0,
            'gss_temp_grad_cos_count': 0,
            'gss_temp_grad_cos_max': 0.0,
            'gss_temp_current_task_chunks': 0,
            'gss_temp_historical_task_chunks': 0,
            'gss_temp_historical_fallback_count': 0
        }
        # make initial set
        if self.init_size > 0:
            if bool(self.class_balance):
                class_size = []
                base_size = int(self.init_size // self.model_params['num_class'])
                for i in range(self.model_params['num_class']):
                    class_size.append(base_size)
                res = self.init_size - self.model_params['num_class'] * base_size
                for i in range(self.model_params['num_class']):
                    if res == 0:
                        break
                    class_size[i] = class_size[i] + 1
                    res -= 1
                init_ids = set()
                for i in range(self.model_params['num_class']):
                    cids = all_class_ids[i]
                    init_cids = random.sample(list(cids), class_size[i])
                    for d_id in init_cids:
                        init_ids.add(d_id)
            else:
                init_ids = random.sample(full_ids, self.init_size)
                init_ids = set(init_ids)
            init_data = get_subset_by_id(
                x=x, y=y, ids=init_ids,
                transforms=self.to_pil if self.transforms is not None else None,
                id_list=id_list)
            for di in init_data:
                d_id = int(di[0])
                lab = int(di[2])
                class_ids[lab].add(d_id)
            with open(self.cur_train_file, 'wb') as fw:
                for di in init_data:
                    pickle.dump(di, fw)
            for d_id in init_ids:
                all_selected_ids.add(d_id)
            if self.save_checkpoint:
                self.dump_selected_ids(selected_ids=all_selected_ids)
            result = train_methods_for_selection.train_model(
                local_path=self.local_path,
                model=init_model,
                train_loader=train_loader,
                train_params=self.train_params,
                eval_loader=full_train_loader,
                eval_mode=self.eval_mode,
                verbose=verbose,
                load_best=False,
                eval_steps=self.eval_steps
            )
            init_model = result
        while len(all_selected_ids) < select_size:
            remaining_select_size = select_size - len(all_selected_ids)
            if self.selection_strategy in ['rel_ccrf', 'rel_gss_temp']:
                cur_incremental_size = resolve_ccrf_effective_chunk_size(
                    selection_chunk_size=self.selection_chunk_size,
                    base_incremental_size=base_incremental_size,
                    remaining_select_size=remaining_select_size,
                    is_current_task=ccrf_is_current_task
                )
            elif self.selection_chunk_size > 0:
                cur_incremental_size = min(self.selection_chunk_size, remaining_select_size)
            else:
                cur_incremental_size = min(base_incremental_size, remaining_select_size)
            id_pool = set()
            for d_id in full_ids:
                if bool(self.only_new_data):
                    if d_id not in all_selected_ids:
                        if self.selection_strategy not in rejection_filter_strategies or d_id not in rejected_ids:
                            id_pool.add(d_id)
                else:
                    if self.selection_strategy not in rejection_filter_strategies or d_id not in rejected_ids:
                        id_pool.add(d_id)
            if len(id_pool) == 0:
                raise ValueError('No candidate samples remain for selection')
            force_select_all_remaining = False
            if self.selection_strategy == 'rel_filter' and len(id_pool) <= remaining_select_size:
                cur_incremental_size = len(id_pool)
                force_select_all_remaining = True
            if bool(self.class_balance):
                class_sizes = make_class_sizes(
                    class_ids=class_ids,
                    incremental_size=cur_incremental_size
                )
            else:
                class_sizes = None
            rand_data = get_subset_by_id(
                x=x,
                y=y,
                ids=id_pool,
                transforms=self.to_pil if self.transforms is not None else None,
                id_list=id_list,
                id2logit=id2logit
            )
            if self.selection_strategy == 'rel':
                selected_data, _ = coreset_selection_functions.select_by_loss_diff(
                    ref_loss_dic=ref_loss_dic,
                    rand_data=rand_data,
                    model=init_model,
                    incremental_size=cur_incremental_size,
                    transforms=self.transforms,
                    on_cuda=self.train_params['use_cuda'],
                    loss_params=self.train_params['loss_params'],
                    class_sizes=class_sizes
                )
            elif self.selection_strategy == 'rel_diversity':
                feature_model = self.ref_model if self.div_feature_source == 'holdout_model' else init_model
                if feature_model is None:
                    feature_model = init_model
                selected_data, _ = coreset_selection_functions.select_by_loss_diff_with_diversity(
                    ref_loss_dic=ref_loss_dic,
                    rand_data=rand_data,
                    model=init_model,
                    incremental_size=cur_incremental_size,
                    transforms=self.transforms,
                    on_cuda=self.train_params['use_cuda'],
                    loss_params=self.train_params['loss_params'],
                    class_sizes=class_sizes,
                    feature_model=feature_model,
                    div_lambda=self.div_lambda,
                    div_candidate_ratio=self.div_candidate_ratio,
                    div_feature_source=self.div_feature_source,
                    div_feature_layer=self.div_feature_layer,
                    div_verbose=self.div_verbose,
                    selected_features_bank=div_selected_features_bank,
                    selected_ids_bank=div_selected_ids_bank,
                    stateful_diag=div_stateful_diag
                )
            elif self.selection_strategy == 'rel_filter':
                feature_model = self.ref_model if self.filter_feature_source == 'holdout_model' else init_model
                if feature_model is None:
                    feature_model = init_model
                if force_select_all_remaining:
                    selected_data, _ = coreset_selection_functions.select_by_loss_diff(
                        ref_loss_dic=ref_loss_dic,
                        rand_data=rand_data,
                        model=init_model,
                        incremental_size=cur_incremental_size,
                        transforms=self.transforms,
                        on_cuda=self.train_params['use_cuda'],
                        loss_params=self.train_params['loss_params'],
                        class_sizes=class_sizes
                    )
                else:
                    selected_data, _, rejected_this_chunk = \
                        coreset_selection_functions.select_by_loss_diff_with_filter(
                            ref_loss_dic=ref_loss_dic,
                            rand_data=rand_data,
                            model=init_model,
                            incremental_size=cur_incremental_size,
                            transforms=self.transforms,
                            on_cuda=self.train_params['use_cuda'],
                            loss_params=self.train_params['loss_params'],
                            class_sizes=class_sizes,
                            feature_model=feature_model,
                            filter_sim_threshold=self.filter_sim_threshold,
                            filter_feature_source=self.filter_feature_source,
                            filter_feature_layer=self.filter_feature_layer,
                            filter_reject_dominated=self.filter_reject_dominated,
                            filter_verbose=self.filter_verbose,
                            rejected_total_before=len(rejected_ids),
                            max_rejections_this_chunk=max(max_rejected_total - len(rejected_ids), 0)
                        )
                    for d_id in rejected_this_chunk:
                        rejected_ids.add(int(d_id))
            elif self.selection_strategy == 'rel_ccrf':
                feature_model = self.ref_model if self.filter_feature_source == 'holdout_model' else init_model
                if feature_model is None:
                    feature_model = init_model
                selected_data, _, rejected_this_chunk, ccrf_chunk_stats = \
                    coreset_selection_functions.select_by_loss_diff_with_ccrf(
                        ref_loss_dic=ref_loss_dic,
                        rand_data=rand_data,
                        model=init_model,
                        incremental_size=cur_incremental_size,
                        transforms=self.transforms,
                        on_cuda=self.train_params['use_cuda'],
                        loss_params=self.train_params['loss_params'],
                        class_sizes=class_sizes,
                        feature_model=feature_model,
                        ccrf_sim_threshold=self.ccrf_sim_threshold,
                        is_current_task=ccrf_is_current_task
                    )
                for d_id in rejected_this_chunk:
                    rejected_ids.add(int(d_id))
                ccrf_stats['ccrf_chunks_total'] += 1
                ccrf_stats['ccrf_rejected_total'] += int(ccrf_chunk_stats['rejected_count'])
                ccrf_stats['ccrf_kept_total'] += int(ccrf_chunk_stats['kept_count'])
                ccrf_stats['ccrf_same_class_pair_checked_count'] += \
                    int(ccrf_chunk_stats['same_class_pair_checked_count'])
                ccrf_stats['ccrf_same_class_pair_rejected_count'] += \
                    int(ccrf_chunk_stats['same_class_pair_rejected_count'])
                ccrf_stats['ccrf_cross_class_pair_ignored_count'] += \
                    int(ccrf_chunk_stats['cross_class_pair_ignored_count'])
                if ccrf_is_current_task:
                    ccrf_stats['ccrf_current_task_chunks'] += 1
                else:
                    ccrf_stats['ccrf_historical_task_chunks'] += 1
                    ccrf_stats['ccrf_historical_fallback_count'] += 1
                if int(ccrf_chunk_stats['rejected_count']) > 0:
                    ccrf_stats['ccrf_chunks_with_rejection'] += 1
            elif self.selection_strategy == 'rel_gss_temp':
                selected_data, _, deferred_this_chunk, gss_chunk_stats = \
                    coreset_selection_functions.select_by_loss_diff_with_gss_temp(
                        ref_loss_dic=ref_loss_dic,
                        rand_data=rand_data,
                        model=init_model,
                        incremental_size=cur_incremental_size,
                        transforms=self.transforms,
                        on_cuda=self.train_params['use_cuda'],
                        loss_params=self.train_params['loss_params'],
                        class_sizes=class_sizes,
                        gss_grad_sim_threshold=self.gss_grad_sim_threshold,
                        gss_grad_layer=self.gss_grad_layer,
                        is_current_task=ccrf_is_current_task
                    )
                gss_temp_stats['gss_temp_chunks_total'] += 1
                gss_temp_stats['gss_temp_deferred_total'] += int(gss_chunk_stats['deferred_count'])
                gss_temp_stats['gss_temp_kept_total'] += int(gss_chunk_stats['kept_count'])
                gss_temp_stats['gss_temp_grad_pair_checked_count'] += \
                    int(gss_chunk_stats['grad_pair_checked_count'])
                gss_temp_stats['gss_temp_grad_pair_deferred_count'] += \
                    int(gss_chunk_stats['grad_pair_deferred_count'])
                if len(str(gss_chunk_stats.get('grad_layer_used', ''))) > 0:
                    gss_temp_stats['gss_temp_grad_layer_used'] = \
                        str(gss_chunk_stats.get('grad_layer_used', ''))
                gss_temp_stats['gss_temp_grad_cos_sum'] += float(gss_chunk_stats['grad_cos_sum'])
                gss_temp_stats['gss_temp_grad_cos_count'] += int(gss_chunk_stats['grad_cos_count'])
                gss_temp_stats['gss_temp_grad_cos_max'] = max(
                    float(gss_temp_stats['gss_temp_grad_cos_max']),
                    float(gss_chunk_stats['grad_cos_max'])
                )
                if ccrf_is_current_task:
                    gss_temp_stats['gss_temp_current_task_chunks'] += 1
                else:
                    gss_temp_stats['gss_temp_historical_task_chunks'] += 1
                    gss_temp_stats['gss_temp_historical_fallback_count'] += 1
                if int(gss_chunk_stats['deferred_count']) > 0:
                    gss_temp_stats['gss_temp_chunks_with_deferral'] += 1
            else:
                raise ValueError('Invalid selection strategy: ' + str(self.selection_strategy))
            flg_add = False
            for di in selected_data:
                d_id = int(di[0])
                lab = int(di[2])
                if d_id not in all_selected_ids:
                    all_selected_ids.add(d_id)
                    flg_add = True
                class_ids[lab].add(d_id)
            if flg_add:
                coreset_selection_functions.add_new_data(data_file=self.cur_train_file, new_data=selected_data)
                cur_train_dataset.load_data()
                remove_ids = set()
                for di in selected_data:
                    d_id = int(di[0])
                    remove_ids.add(d_id)
                if full_train_loader is not None:
                    full_train_loader.dataset.remove_data_by_id(ids=remove_ids)
                init_model = utils.build_model(model_params=self.model_params)
                if self.save_checkpoint:
                    self.dump_selected_ids(selected_ids=all_selected_ids)
            result = train_methods_for_selection.train_model(
                local_path=self.local_path,
                model=init_model,
                train_loader=train_loader,
                train_params=self.train_params,
                eval_loader=full_train_loader,
                eval_mode=self.eval_mode,
                verbose=verbose,
                load_best=False,
                eval_steps=self.eval_steps
            )
            trained_model = result
            init_model = trained_model
            print('finish selecting samples:', len(all_selected_ids))
        if len(full_data_file) > 0:
            os.remove(full_data_file)
        # get selected data
        selected_data = []
        with open(self.cur_train_file, 'rb') as fr:
            while True:
                try:
                    di = pickle.load(fr)
                    selected_data.append(di)
                except EOFError:
                    break
        if self.selection_strategy == 'rel_diversity' and self.div_verbose:
            selected_ids = []
            selected_data_structure_ok = True
            for di in selected_data:
                if not isinstance(di, list) or len(di) not in [3, 4]:
                    selected_data_structure_ok = False
                    continue
                if di[0] is None or di[1] is None or di[2] is None:
                    selected_data_structure_ok = False
                selected_ids.append(int(di[0]))
            duplicate_id_count = len(selected_ids) - len(set(selected_ids))
            print('[RD-CHECK] stateful_rd=True')
            print('[RD-CHECK] total_selected>=20=' + str(len(selected_data) >= 20))
            print('[RD-CHECK] selection_size_each_call=1=' +
                  str(len(div_stateful_diag['selection_sizes']) > 0 and
                      all(x == 1 for x in div_stateful_diag['selection_sizes'])))
            print('[RD-CHECK] selected_bank_size_final>=20=' + str(len(div_selected_ids_bank) >= 20))
            print('[RD-CHECK] max_similarity_nonzero_after_first=' +
                  str(div_stateful_diag['max_similarity_nonzero_after_first']))
            print('[RD-CHECK] div_penalty_active=' + str(div_stateful_diag['div_penalty_active']))
            print('[RD-CHECK] duplicate_id_count=' + str(duplicate_id_count))
            print('[RD-CHECK] selected_data_structure_ok=' +
                  str(selected_data_structure_ok and duplicate_id_count == 0))
        if self.selection_strategy == 'rel_filter' and self.filter_verbose:
            selected_ids = []
            selected_data_structure_ok = True
            for di in selected_data:
                if not isinstance(di, list) or len(di) not in [3, 4]:
                    selected_data_structure_ok = False
                    continue
                if di[0] is None or di[1] is None or di[2] is None:
                    selected_data_structure_ok = False
                selected_ids.append(int(di[0]))
            duplicate_id_count = len(selected_ids) - len(set(selected_ids))
            rejected_intersection_selected_count = len(rejected_ids.intersection(set(selected_ids)))
            print('[RF-CHECK] selected_data_structure_ok=' +
                  str(selected_data_structure_ok and duplicate_id_count == 0))
            print('[RF-CHECK] duplicate_selected_id_count=' + str(duplicate_id_count))
            print('[RF-CHECK] rejected_intersection_selected_count=' +
                  str(rejected_intersection_selected_count))
            print('[RF-CHECK] rejected_scope=current_incremental_selection')
        if self.selection_strategy == 'rel_ccrf':
            if ccrf_stats['ccrf_chunks_total'] > 0:
                ccrf_mean_kept_per_chunk = \
                    float(ccrf_stats['ccrf_kept_total']) / float(ccrf_stats['ccrf_chunks_total'])
            else:
                ccrf_mean_kept_per_chunk = 0.0
            print('[CCRF-SUMMARY] ccrf_enabled=' + str(ccrf_stats['ccrf_enabled']))
            print('[CCRF-SUMMARY] ccrf_sim_threshold=' + str(ccrf_stats['ccrf_sim_threshold']))
            print('[CCRF-SUMMARY] ccrf_chunk_size=' + str(ccrf_stats['ccrf_chunk_size']))
            print('[CCRF-SUMMARY] ccrf_chunks_total=' + str(ccrf_stats['ccrf_chunks_total']))
            print('[CCRF-SUMMARY] ccrf_chunks_with_rejection=' +
                  str(ccrf_stats['ccrf_chunks_with_rejection']))
            print('[CCRF-SUMMARY] ccrf_rejected_total=' + str(ccrf_stats['ccrf_rejected_total']))
            print('[CCRF-SUMMARY] ccrf_kept_total=' + str(ccrf_stats['ccrf_kept_total']))
            print('[CCRF-SUMMARY] ccrf_same_class_pair_checked_count=' +
                  str(ccrf_stats['ccrf_same_class_pair_checked_count']))
            print('[CCRF-SUMMARY] ccrf_same_class_pair_rejected_count=' +
                  str(ccrf_stats['ccrf_same_class_pair_rejected_count']))
            print('[CCRF-SUMMARY] ccrf_cross_class_pair_ignored_count=' +
                  str(ccrf_stats['ccrf_cross_class_pair_ignored_count']))
            print('[CCRF-SUMMARY] ccrf_mean_kept_per_chunk=%.6f' % ccrf_mean_kept_per_chunk)
            print('[CCRF-SUMMARY] ccrf_current_task_chunks=' +
                  str(ccrf_stats['ccrf_current_task_chunks']))
            print('[CCRF-SUMMARY] ccrf_historical_task_chunks=' +
                  str(ccrf_stats['ccrf_historical_task_chunks']))
            print('[CCRF-SUMMARY] ccrf_historical_fallback_enabled=' +
                  str(ccrf_stats['ccrf_historical_fallback_enabled']))
            print('[CCRF-SUMMARY] ccrf_historical_fallback_count=' +
                  str(ccrf_stats['ccrf_historical_fallback_count']))
            print('[CCRF-SUMMARY] ccrf_effective_chunk_size_current=' +
                  str(ccrf_stats['ccrf_effective_chunk_size_current']))
            print('[CCRF-SUMMARY] ccrf_effective_chunk_size_historical=' +
                  str(ccrf_stats['ccrf_effective_chunk_size_historical']))
        if self.selection_strategy == 'rel_gss_temp':
            if gss_temp_stats['gss_temp_chunks_total'] > 0:
                gss_temp_mean_kept_per_chunk = \
                    float(gss_temp_stats['gss_temp_kept_total']) / \
                    float(gss_temp_stats['gss_temp_chunks_total'])
            else:
                gss_temp_mean_kept_per_chunk = 0.0
            if gss_temp_stats['gss_temp_grad_cos_count'] > 0:
                gss_temp_grad_cos_mean = \
                    float(gss_temp_stats['gss_temp_grad_cos_sum']) / \
                    float(gss_temp_stats['gss_temp_grad_cos_count'])
            else:
                gss_temp_grad_cos_mean = 0.0
            print('[GSS-TEMP-SUMMARY] gss_temp_enabled=' +
                  str(gss_temp_stats['gss_temp_enabled']))
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_sim_threshold=' +
                  str(gss_temp_stats['gss_temp_grad_sim_threshold']))
            print('[GSS-TEMP-SUMMARY] gss_temp_chunk_size=' +
                  str(gss_temp_stats['gss_temp_chunk_size']))
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_layer=' +
                  str(gss_temp_stats['gss_temp_grad_layer']))
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_layer_used=' +
                  str(gss_temp_stats['gss_temp_grad_layer_used']))
            print('[GSS-TEMP-SUMMARY] gss_temp_chunks_total=' +
                  str(gss_temp_stats['gss_temp_chunks_total']))
            print('[GSS-TEMP-SUMMARY] gss_temp_chunks_with_deferral=' +
                  str(gss_temp_stats['gss_temp_chunks_with_deferral']))
            print('[GSS-TEMP-SUMMARY] gss_temp_deferred_total=' +
                  str(gss_temp_stats['gss_temp_deferred_total']))
            print('[GSS-TEMP-SUMMARY] gss_temp_permanent_rejected_total=' +
                  str(gss_temp_stats['gss_temp_permanent_rejected_total']))
            print('[GSS-TEMP-SUMMARY] gss_temp_kept_total=' +
                  str(gss_temp_stats['gss_temp_kept_total']))
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_pair_checked_count=' +
                  str(gss_temp_stats['gss_temp_grad_pair_checked_count']))
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_pair_deferred_count=' +
                  str(gss_temp_stats['gss_temp_grad_pair_deferred_count']))
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_cos_mean=%.6f' %
                  gss_temp_grad_cos_mean)
            print('[GSS-TEMP-SUMMARY] gss_temp_grad_cos_max=%.6f' %
                  float(gss_temp_stats['gss_temp_grad_cos_max']))
            print('[GSS-TEMP-SUMMARY] gss_temp_mean_kept_per_chunk=%.6f' %
                  gss_temp_mean_kept_per_chunk)
            print('[GSS-TEMP-SUMMARY] gss_temp_current_task_chunks=' +
                  str(gss_temp_stats['gss_temp_current_task_chunks']))
            print('[GSS-TEMP-SUMMARY] gss_temp_historical_task_chunks=' +
                  str(gss_temp_stats['gss_temp_historical_task_chunks']))
            print('[GSS-TEMP-SUMMARY] gss_temp_historical_fallback_count=' +
                  str(gss_temp_stats['gss_temp_historical_fallback_count']))
        return selected_data

    def clear_path(self):
        if os.path.exists(self.cur_train_file):
            os.remove(self.cur_train_file)

    def reset_ref_model(self):
        self.ref_model = None

    def dump_selected_ids(self, selected_ids):
        num_sps = len(selected_ids)
        dump_file = os.path.join(self.local_path, 'selected_ids_' + str(num_sps) + '.pkl')
        with open(dump_file, 'wb') as fw:
            pickle.dump(selected_ids, fw)


def resolve_ccrf_effective_chunk_size(selection_chunk_size, base_incremental_size, remaining_select_size,
                                      is_current_task):
    if not is_current_task:
        return min(1, remaining_select_size)
    if selection_chunk_size > 0:
        return min(selection_chunk_size, remaining_select_size)
    return min(base_incremental_size, remaining_select_size)


def get_class_dic(y):
    class_dic = {}
    for i in range(y.shape[0]):
        lab = int(y[i])
        if lab not in class_dic:
            class_dic[lab] = [i]
        else:
            class_dic[lab].append(i)
    return class_dic


def get_subset_by_id(x, y, ids, transforms=None, id_list=None, id2logit=None):
    selected_data = []
    if id_list is None:
        d_pos = ids
    else:
        d_pos = []
        id_pool = set(ids)
        for i, d_id in enumerate(id_list):
            if d_id in id_pool:
                d_pos.append(i)
    for pi in d_pos:
        if transforms is None:
            sp = torch.tensor(x[pi], dtype=torch.float32).clone().detach()
        else:
            sp = transforms(torch.tensor(x[pi], dtype=torch.float32).clone().detach())
        if id_list is None:
            d_id = pi
        else:
            d_id = id_list[pi]
        data = [d_id, sp, int(y[pi])]
        if id2logit is not None:
            data.append(id2logit[d_id])
        selected_data.append(data)
    return selected_data


def make_class_sizes(class_ids, incremental_size):
    class_cnts = {}
    max_cnt = -1
    for ci in class_ids.keys():
        class_cnts[ci] = len(class_ids[ci])
        if len(class_ids[ci]) > max_cnt:
            max_cnt = len(class_ids[ci])
    sorted_cnt = sorted(class_cnts.items(), key=lambda x: x[1])
    class_sizes = {}
    for ci in class_ids.keys():
        class_sizes[ci] = 0
    rest_size = incremental_size
    for i in range(len(sorted_cnt)):
        ci, cnt = sorted_cnt[i]
        to_select = max_cnt - cnt
        to_select = min(to_select, rest_size)
        class_sizes[ci] = to_select
        rest_size -= to_select
        if rest_size == 0:
            break
    if rest_size > 0:
        base_size = int(rest_size // len(class_ids))
        for ci in class_sizes.keys():
            class_sizes[ci] = class_sizes[ci] + base_size
        rest_size = rest_size - base_size * len(class_ids)
        for ci in class_sizes.keys():
            class_sizes[ci] += 1
            rest_size -= 1
            if rest_size == 0:
                break
    return class_sizes
