# -*-coding:utf8-*-

import torch
from torch.utils.data import DataLoader
import torchvision
import pickle
import os
import random
import copy
import numpy as np

import utils
from dataset import single_task_dataset
from coreset_selection import coreset_selection_functions
from coreset_selection import train_methods_for_selection
from functions import train_methods
from functions import loss_functions


class RhoSelectionAgent(object):
    def __init__(self, local_path, transforms, init_size, selection_steps, cur_train_lr, cur_train_steps, use_cuda,
                 eval_mode, early_stop, eval_steps, model_params, ref_train_params, seed, ref_model=None,
                 class_balance=True, only_new_data=True, loss_params=None, save_checkpoint=False,
                 selection_strategy='rel', selection_chunk_size=0, div_lambda=0.1, div_candidate_ratio=3,
                 div_feature_source='current_model', div_feature_layer='penultimate', div_verbose=False,
                  filter_sim_threshold=0.85, filter_feature_source='current_model',
                  filter_feature_layer='penultimate', filter_reject_dominated=False, filter_verbose=False,
                  gss_anchor_replace_window=8, gss_anchor_replace_anchor_size=4,
                  gss_anchor_replace_sim_threshold=0.90, gss_grad_layer='classifier',
                  gss_anchor_replace_prob_seed_offset=0,
                  gss_anchor_replace_prob_conservativeness=1.0):
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
        self.filter_feature_source = filter_feature_source
        self.filter_feature_layer = filter_feature_layer
        self.filter_reject_dominated = filter_reject_dominated
        self.filter_verbose = filter_verbose
        self.gss_anchor_replace_window = gss_anchor_replace_window
        self.gss_anchor_replace_anchor_size = gss_anchor_replace_anchor_size
        self.gss_anchor_replace_sim_threshold = gss_anchor_replace_sim_threshold
        self.gss_grad_layer = gss_grad_layer
        self.gss_anchor_replace_prob_seed_offset = gss_anchor_replace_prob_seed_offset
        self.gss_anchor_replace_prob_conservativeness = float(gss_anchor_replace_prob_conservativeness)
        base_prob_seed = 0 if seed is None else int(seed)
        prob_seed = (base_prob_seed + int(gss_anchor_replace_prob_seed_offset)) % (2 ** 32)
        self.gss_anchor_replace_prob_rng = np.random.RandomState(prob_seed)
        self.gss_anchor_replace_stats = coreset_selection_functions.make_gss_anchor_replace_stats()
        self.gss_iqp_stats = coreset_selection_functions.make_gss_iqp_stats()
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

    def _get_ref_loss_params(self, ideal_logit=False):
        if ideal_logit:
            return {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            }
        return self.train_params['loss_params']

    def _restore_model_status(self, model, train_status, was_cuda):
        if was_cuda and torch.cuda.is_available():
            model.cuda()
        else:
            model.cpu()
        model.train(train_status)

    def compute_ref_loss_dic_from_model(self, x, y, ref_proxy_model, id_list=None, id2logit=None,
                                        ideal_logit=False, loss_dic_dump_file=None,
                                        temp_fname='temp_ref_proxy_data.pkl'):
        if ref_proxy_model is None:
            raise ValueError('ref_proxy_model is required for rel_ref_mode=final_model')
        temp_loader, temp_file = self.make_data_loader(
            x=x,
            y=y,
            fname=temp_fname,
            batch_size=20,
            id_list=id_list,
            id2logit=id2logit
        )
        train_status = ref_proxy_model.training
        try:
            first_param = next(ref_proxy_model.parameters())
            was_cuda = first_param.is_cuda
        except StopIteration:
            was_cuda = False
        try:
            ref_loss_dic = utils.compute_loss_dic(
                ref_model=ref_proxy_model,
                data_loader=temp_loader,
                aug_iters=1,
                use_cuda=self.train_params['use_cuda'],
                loss_params=self._get_ref_loss_params(ideal_logit=ideal_logit)
            )
            if loss_dic_dump_file is not None:
                with open(loss_dic_dump_file, 'wb') as fw:
                    pickle.dump(ref_loss_dic, fw)
            return ref_loss_dic
        finally:
            self._restore_model_status(ref_proxy_model, train_status, was_cuda)
            if hasattr(temp_loader.dataset, 'remove_shuffle_file'):
                temp_loader.dataset.remove_shuffle_file()
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def _compute_loss_correct_true_prob_dic(self, ref_model, data_loader, loss_params):
        train_status = ref_model.training
        try:
            first_param = next(ref_model.parameters())
            was_cuda = first_param.is_cuda
        except StopIteration:
            was_cuda = False
        ref_model.eval()
        if self.train_params['use_cuda']:
            ref_model.cuda()
        loss_fn = loss_functions.CompliedLoss(
            ce_factor=loss_params['ce_factor'],
            mse_factor=loss_params['mse_factor'],
            reduction='none'
        )
        loss_dic = {}
        correct_dic = {}
        true_prob_dic = {}
        try:
            with torch.no_grad():
                for data in data_loader:
                    if len(data) == 4:
                        d_ids, sps, labs, logit = data
                    else:
                        d_ids, sps, labs = data
                        logit = None
                    if self.train_params['use_cuda']:
                        sps = sps.cuda()
                        labs = labs.cuda()
                        if logit is not None:
                            logit = logit.cuda()
                    out = ref_model(sps)
                    loss = loss_fn(out, labs, logit)
                    true_prob = torch.softmax(out, dim=1).gather(1, labs.view(-1, 1)).view(-1)
                    pred = out.argmax(dim=1)
                    correct = pred.eq(labs).clone().detach()
                    if self.train_params['use_cuda']:
                        loss = loss.cpu()
                        correct = correct.cpu()
                        true_prob = true_prob.cpu()
                    loss = loss.clone().detach().numpy()
                    correct = correct.numpy()
                    true_prob = true_prob.clone().detach().numpy()
                    for j in range(sps.shape[0]):
                        d_id = int(d_ids[j].numpy())
                        loss_dic[d_id] = float(loss[j])
                        correct_dic[d_id] = float(correct[j])
                        true_prob_dic[d_id] = float(true_prob[j])
        finally:
            self._restore_model_status(ref_model, train_status, was_cuda)
        return loss_dic, correct_dic, true_prob_dic

    def _print_snapshot_dynamic_calibration_summary(self, diagnostics):
        def stat_line(name, values):
            values = np.asarray(values, dtype=np.float32)
            if values.size == 0:
                values = np.asarray([0.0], dtype=np.float32)
            print('[snapshot dynamic calibration] ' + name + ' mean/min/max %.6f %.6f %.6f' %
                  (float(np.mean(values)), float(np.min(values)), float(np.max(values))))

        print('[snapshot dynamic calibration]')
        print('[snapshot dynamic calibration] alpha=%.6f beta=%.6f rho_max=%.6f tau=%.6f' %
              (float(diagnostics['alpha']), float(diagnostics['beta']),
               float(diagnostics['rho_max']), float(diagnostics['tau'])))
        print('[snapshot dynamic calibration] norm_scope=' + str(diagnostics['norm_scope']))
        print('[snapshot dynamic calibration] ablation_mode=' + str(bool(diagnostics['ablation_mode'])))
        print('[snapshot dynamic calibration] use_confidence=' + str(bool(diagnostics['use_confidence'])))
        print('[snapshot dynamic calibration] use_variability=' + str(bool(diagnostics['use_variability'])))
        print('[snapshot dynamic calibration] use_forgetting=' + str(bool(diagnostics['use_forgetting'])))
        print('[snapshot dynamic calibration] snapshot_calib_conf_gate=' +
              str(diagnostics['conf_gate']))
        print('[snapshot dynamic calibration] snapshot_calib_conf_gamma=%.6f' %
              float(diagnostics['conf_gamma']))
        print('[snapshot dynamic calibration] snapshot_calib_conf_eps=%.6g' %
              float(diagnostics['conf_eps']))
        print('[snapshot dynamic calibration] snapshot_calib_conf_log_lambda=%.6f' %
              float(diagnostics['conf_log_lambda']))
        print('[snapshot dynamic calibration] conf_gate=' + str(diagnostics['conf_gate']))
        print('[snapshot dynamic calibration] conf_log_lambda=%.6f' %
              float(diagnostics['conf_log_lambda']))
        print('[snapshot dynamic calibration] snapshot_calib_lowconf_q=%.6f' %
              float(diagnostics['lowconf_q']))
        print('[snapshot dynamic calibration] snapshot_calib_lowvar_tail_q=%.6f' %
              float(diagnostics['lowvar_tail_q']))
        print('[snapshot dynamic calibration] snapshot_calib_lowvar_eta=%.6f' %
              float(diagnostics['lowvar_eta']))
        print('[snapshot dynamic calibration] highconf_protect_q=%.6f' %
              float(diagnostics['highconf_protect_q']))
        print('[snapshot dynamic calibration] lowconf_count=%d' %
              int(diagnostics['lowconf_count']))
        print('[snapshot dynamic calibration] lowvar_tail_count=%d' %
              int(diagnostics['lowvar_tail_count']))
        print('[snapshot dynamic calibration] joint_lowconf_lowvar_count=%d' %
              int(diagnostics['joint_lowconf_lowvar_count']))
        print('[snapshot dynamic calibration] highconf_protected_count=%d' %
              int(diagnostics['highconf_protected_count']))
        print('[snapshot dynamic calibration] not_highconf_count=%d' %
              int(diagnostics['not_highconf_count']))
        print('[snapshot dynamic calibration] active_protected_lowvar_count=%d' %
              int(diagnostics['active_protected_lowvar_count']))
        for key in ['conf', 'var', 'var_norm', 'forg',
                    'conf_risk', 'var_risk', 'forg_risk', 'final risk', 'rho',
                    'q_alpha', 'q_beta', 'ref_loss']:
            stat_line(key, diagnostics.get(key, []))
        for key in ['lowvar_tail_residual', 'gate_llr', 'gate_final']:
            values = np.asarray(diagnostics.get(key, []), dtype=np.float32)
            if values.size == 0:
                values = np.asarray([0.0], dtype=np.float32)
            print('[snapshot dynamic calibration] ' + key + ' mean/max %.6f %.6f' %
                  (float(np.mean(values)), float(np.max(values))))
        values = np.asarray(
            diagnostics.get('protected_lowvar_residual', []), dtype=np.float32)
        if values.size == 0:
            values = np.asarray([0.0], dtype=np.float32)
        print('[snapshot dynamic calibration] protected_lowvar_residual mean/max %.6f %.6f' %
              (float(np.mean(values)), float(np.max(values))))

    def compute_ref_loss_dic_from_snapshots(self, x, y, snapshot_state_dicts, id_list=None, id2logit=None,
                                            snapshot_reduce='quantile', snapshot_quantile=0.2,
                                            snapshot_agreement_lambda=0.0, snapshot_dynamic_calibration=False,
                                            snapshot_calib_beta=0.2, snapshot_calib_rho_max=0.1,
                                            snapshot_calib_tau=0.5, snapshot_calib_norm_scope='task',
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
                                            ideal_logit=False, loss_dic_dump_file=None):
        if snapshot_state_dicts is None or len(snapshot_state_dicts) == 0:
            raise ValueError('snapshot_state_dicts is required for rel_ref_mode=snapshot')
        if bool(snapshot_dynamic_calibration):
            if float(snapshot_agreement_lambda) > 0.0:
                raise ValueError('snapshot_dynamic_calibration cannot be combined with snapshot_agreement_lambda > 0')
            if snapshot_reduce != 'quantile':
                raise ValueError('snapshot_dynamic_calibration requires snapshot_reduce=quantile')
        temp_loader, temp_file = self.make_data_loader(
            x=x,
            y=y,
            fname='temp_snapshot_proxy_data.pkl',
            batch_size=20,
            id_list=id_list,
            id2logit=id2logit
        )
        loss_params = self._get_ref_loss_params(ideal_logit=ideal_logit)
        id2losses = {}
        id2corrects = {}
        id2p_trues = {}
        try:
            for snapshot_idx, state_dict in enumerate(snapshot_state_dicts):
                snapshot_model = utils.build_model(model_params=self.model_params)
                snapshot_model.load_state_dict(state_dict)
                loss_dic, correct_dic, true_prob_dic = self._compute_loss_correct_true_prob_dic(
                    ref_model=snapshot_model,
                    data_loader=temp_loader,
                    loss_params=loss_params
                )
                for d_id in loss_dic.keys():
                    id2losses.setdefault(d_id, []).append(float(loss_dic[d_id]))
                    id2corrects.setdefault(d_id, []).append(float(correct_dic[d_id]))
                    id2p_trues.setdefault(d_id, []).append(float(true_prob_dic[d_id]))
                if snapshot_verbose:
                    print('[SNAPSHOT-REF] processed snapshot', snapshot_idx + 1, 'of', len(snapshot_state_dicts))
                del snapshot_model
            if bool(snapshot_dynamic_calibration):
                ref_loss_dic, diagnostics = \
                    coreset_selection_functions.compute_dynamic_calibrated_snapshot_ref_loss_dic(
                        id2losses=id2losses,
                        id2p_trues=id2p_trues,
                        id2corrects=id2corrects,
                        alpha=snapshot_quantile,
                        beta=snapshot_calib_beta,
                        rho_max=snapshot_calib_rho_max,
                        tau=snapshot_calib_tau,
                        norm_scope=snapshot_calib_norm_scope,
                        use_confidence=snapshot_calib_use_confidence,
                        use_variability=snapshot_calib_use_variability,
                        use_forgetting=snapshot_calib_use_forgetting,
                        conf_gate=snapshot_calib_conf_gate,
                        conf_gamma=snapshot_calib_conf_gamma,
                        conf_eps=snapshot_calib_conf_eps,
                        conf_log_lambda=snapshot_calib_conf_log_lambda,
                        lowconf_q=snapshot_calib_lowconf_q,
                        lowvar_tail_q=snapshot_calib_lowvar_tail_q,
                        lowvar_eta=snapshot_calib_lowvar_eta,
                        highconf_protect_q=snapshot_calib_highconf_protect_q
                    )
                self._print_snapshot_dynamic_calibration_summary(diagnostics)
            else:
                ref_loss_dic = {}
                for d_id in id2losses.keys():
                    agreement = float(np.mean(id2corrects[d_id]))
                    ref_loss_dic[d_id] = coreset_selection_functions.compute_snapshot_proxy_loss(
                        losses=id2losses[d_id],
                        snapshot_reduce=snapshot_reduce,
                        snapshot_quantile=snapshot_quantile,
                        agreement=agreement,
                        agreement_lambda=snapshot_agreement_lambda
                )
            if loss_dic_dump_file is not None:
                with open(loss_dic_dump_file, 'wb') as fw:
                    pickle.dump(ref_loss_dic, fw)
            if snapshot_verbose:
                print('[SNAPSHOT-REF] ref_loss_dic size', len(ref_loss_dic))
            return ref_loss_dic
        finally:
            if hasattr(temp_loader.dataset, 'remove_shuffle_file'):
                temp_loader.dataset.remove_shuffle_file()
            if os.path.exists(temp_file):
                os.remove(temp_file)

    def incremental_selection(self, x, y, select_size, id_list=None, loss_dic=None, loss_dic_dump_file=None,
                              verbose=True, class_pool=None, id2logit=None, ideal_logit=False, extra_data=None,
                              gss_anchor_replace_is_current_task=True):
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
            cur_incremental_size = coreset_selection_functions.gss_anchor_replace_resolve_chunk_size(
                selection_strategy=self.selection_strategy,
                is_current_task=gss_anchor_replace_is_current_task,
                selection_chunk_size=self.selection_chunk_size,
                base_incremental_size=base_incremental_size,
                remaining_select_size=remaining_select_size
            )
            id_pool = set()
            for d_id in full_ids:
                if bool(self.only_new_data):
                    if d_id not in all_selected_ids:
                        if self.selection_strategy != 'rel_filter' or d_id not in rejected_ids:
                            id_pool.add(d_id)
                else:
                    if self.selection_strategy != 'rel_filter' or d_id not in rejected_ids:
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
            if self.selection_strategy == 'rel' or \
                    (self.selection_strategy == 'rel_gss_anchor_replace' and
                     not gss_anchor_replace_is_current_task) or \
                    coreset_selection_functions.gss_anchor_replace_uses_historical_rel_top4(
                        self.selection_strategy, gss_anchor_replace_is_current_task) or \
                    coreset_selection_functions.gss_iqp_uses_historical_rel_top4(
                        self.selection_strategy, gss_anchor_replace_is_current_task):
                if self.selection_strategy == 'rel_gss_anchor_replace':
                    self.gss_anchor_replace_stats['gss_anchor_replace_historical_task_chunks'] += 1
                    self.gss_anchor_replace_stats['gss_anchor_replace_historical_fallback_count'] += 1
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
                if coreset_selection_functions.gss_anchor_replace_uses_historical_rel_top4(
                        self.selection_strategy, gss_anchor_replace_is_current_task):
                    coreset_selection_functions.record_gss_anchor_replace_historical_top4(
                        self.gss_anchor_replace_stats, len(selected_data))
                if coreset_selection_functions.gss_iqp_uses_historical_rel_top4(
                        self.selection_strategy, gss_anchor_replace_is_current_task):
                    coreset_selection_functions.record_gss_iqp_historical_top4(
                        self.gss_iqp_stats, len(selected_data))
            elif coreset_selection_functions.gss_anchor_replace_uses_anchor(
                    self.selection_strategy, gss_anchor_replace_is_current_task):
                selected_data, _ = coreset_selection_functions.select_by_loss_diff_with_anchor_replace(
                    ref_loss_dic=ref_loss_dic,
                    rand_data=rand_data,
                    model=init_model,
                    incremental_size=cur_incremental_size,
                    transforms=self.transforms,
                    on_cuda=self.train_params['use_cuda'],
                    loss_params=self.train_params['loss_params'],
                    class_sizes=class_sizes,
                    window_size=self.gss_anchor_replace_window,
                    anchor_size=self.gss_anchor_replace_anchor_size,
                    sim_threshold=self.gss_anchor_replace_sim_threshold,
                    grad_layer=self.gss_grad_layer,
                    stats=self.gss_anchor_replace_stats,
                    is_current_task=gss_anchor_replace_is_current_task,
                    probabilistic_replace=(
                        coreset_selection_functions.gss_anchor_replace_uses_prob_replace(
                            self.selection_strategy)),
                    prob_rng=self.gss_anchor_replace_prob_rng,
                    prob_conservativeness=self.gss_anchor_replace_prob_conservativeness
                )
            elif coreset_selection_functions.gss_iqp_uses_iqp(
                    self.selection_strategy, gss_anchor_replace_is_current_task):
                selected_data, _ = coreset_selection_functions.select_by_loss_diff_with_gss_iqp(
                    ref_loss_dic=ref_loss_dic,
                    rand_data=rand_data,
                    model=init_model,
                    incremental_size=cur_incremental_size,
                    transforms=self.transforms,
                    on_cuda=self.train_params['use_cuda'],
                    loss_params=self.train_params['loss_params'],
                    class_sizes=class_sizes,
                    window_size=self.gss_anchor_replace_window,
                    select_size=self.gss_anchor_replace_anchor_size,
                    grad_layer=self.gss_grad_layer,
                    stats=self.gss_iqp_stats,
                    is_current_task=gss_anchor_replace_is_current_task
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
        if coreset_selection_functions.gss_anchor_replace_is_strategy(self.selection_strategy) and \
                gss_anchor_replace_is_current_task:
            self.print_gss_anchor_replace_summary()
        if coreset_selection_functions.gss_iqp_is_strategy(self.selection_strategy) and \
                gss_anchor_replace_is_current_task:
            self.print_gss_iqp_summary()
        return selected_data

    def reset_gss_anchor_replace_summary(self):
        self.gss_anchor_replace_stats = coreset_selection_functions.make_gss_anchor_replace_stats()

    def reset_gss_iqp_summary(self):
        self.gss_iqp_stats = coreset_selection_functions.make_gss_iqp_stats()

    def print_gss_iqp_summary(self):
        stats = self.gss_iqp_stats
        chunks_total = int(stats.get('gss_iqp_chunks_total', 0))
        final_kept_total = int(stats.get('gss_iqp_final_kept_total', 0))
        current_iqp_chunks = int(stats.get('gss_iqp_current_iqp_chunks', 0))
        historical_top4_count = int(stats.get('gss_iqp_historical_top4_count', 0))
        historical_iqp_count = int(stats.get('gss_iqp_historical_iqp_count', 0))
        iqp_changed_chunks = int(stats.get('gss_iqp_changed_chunks', 0))
        iqp_changed_samples = int(stats.get('gss_iqp_changed_samples', 0))
        pairwise_rel_sum = float(stats.get('gss_iqp_pairwise_cos_rel_top4_sum', 0.0))
        pairwise_iqp_sum = float(stats.get('gss_iqp_pairwise_cos_iqp_selected_sum', 0.0))
        rel_top4_sum = float(stats.get('gss_iqp_rel_sum_rel_top4_sum', 0.0))
        rel_iqp_sum = float(stats.get('gss_iqp_rel_sum_iqp_selected_sum', 0.0))
        mean_pairwise_rel = pairwise_rel_sum / max(current_iqp_chunks, 1)
        mean_pairwise_iqp = pairwise_iqp_sum / max(current_iqp_chunks, 1)
        mean_rel_top4 = rel_top4_sum / max(current_iqp_chunks, 1)
        mean_rel_iqp = rel_iqp_sum / max(current_iqp_chunks, 1)
        final_mean = final_kept_total / max(chunks_total, 1)
        print('[GSS-IQP-SUMMARY] GSS-IQP-SUMMARY')
        print('[GSS-IQP-SUMMARY] block_id=' + str(int(stats.get('gss_iqp_block_id', 0))))
        print('[GSS-IQP-SUMMARY] current_iqp_chunks=' + str(current_iqp_chunks))
        print('[GSS-IQP-SUMMARY] historical_top4_count=' + str(historical_top4_count))
        print('[GSS-IQP-SUMMARY] historical_iqp_count=' + str(historical_iqp_count))
        print('[GSS-IQP-SUMMARY] iqp_changed_chunks=' + str(iqp_changed_chunks))
        print('[GSS-IQP-SUMMARY] iqp_changed_samples=' + str(iqp_changed_samples))
        print('[GSS-IQP-SUMMARY] mean_pairwise_cos_rel_top4=%.6f' % float(mean_pairwise_rel))
        print('[GSS-IQP-SUMMARY] mean_pairwise_cos_iqp_selected=%.6f' % float(mean_pairwise_iqp))
        print('[GSS-IQP-SUMMARY] mean_pairwise_cos_delta=%.6f' %
              float(mean_pairwise_iqp - mean_pairwise_rel))
        print('[GSS-IQP-SUMMARY] mean_rel_sum_rel_top4=%.6f' % float(mean_rel_top4))
        print('[GSS-IQP-SUMMARY] mean_rel_sum_iqp_selected=%.6f' % float(mean_rel_iqp))
        print('[GSS-IQP-SUMMARY] mean_rel_sum_drop=%.6f' % float(mean_rel_top4 - mean_rel_iqp))
        print('[GSS-IQP-SUMMARY] final_mean_kept_per_chunk=%.6f' % float(final_mean))
        print('[GSS-IQP-SUMMARY] permanent_rejected_total=0')

    def print_gss_anchor_replace_summary(self):
        stats = self.gss_anchor_replace_stats
        chunks_total = int(stats.get('gss_anchor_replace_chunks_total', 0))
        final_kept_total = int(stats.get('gss_anchor_replace_final_kept_total', 0))
        tail_total = int(stats.get('gss_anchor_replace_tail_candidates_total', 0))
        replaced_total = int(stats.get('gss_anchor_replace_replaced_total', 0))
        skip_total = int(stats.get('gss_anchor_replace_skip_similar_total', 0))
        similarity_sum = float(stats.get('gss_anchor_replace_similarity_sum', 0.0))
        similarity_count = int(stats.get('gss_anchor_replace_similarity_count', 0))
        similarity_max = float(stats.get('gss_anchor_replace_similarity_max', 0.0))
        rel_gap_sum = float(stats.get('gss_anchor_replace_rel_gap_sum', 0.0))
        rel_gap_count = int(stats.get('gss_anchor_replace_rel_gap_count', 0))
        current_task_chunks = int(stats.get('gss_anchor_replace_current_task_chunks', 0))
        historical_task_chunks = int(stats.get('gss_anchor_replace_historical_task_chunks', 0))
        current_anchor_replace_count = int(stats.get('gss_anchor_replace_current_anchor_replace_count', 0))
        historical_anchor_replace_count = int(stats.get('gss_anchor_replace_historical_anchor_replace_count', 0))
        historical_top4_count = int(stats.get('gss_anchor_replace_historical_top4_count', 0))
        historical_fallback_count = int(stats.get('gss_anchor_replace_historical_fallback_count', 0))
        final_mean = final_kept_total / max(chunks_total, 1)
        mean_replace = replaced_total / max(chunks_total, 1)
        similarity_mean = similarity_sum / max(similarity_count, 1)
        rel_gap_mean = rel_gap_sum / max(rel_gap_count, 1)
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_enabled=True')
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_window=' +
              str(self.gss_anchor_replace_window))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_anchor_size=' +
              str(self.gss_anchor_replace_anchor_size))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_sim_threshold=%.6f' %
              float(self.gss_anchor_replace_sim_threshold))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_grad_layer=' +
              str(self.gss_grad_layer))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_grad_layer_used=' +
              str(stats.get('gss_anchor_replace_grad_layer_used', '')))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_chunks_total=' +
              str(chunks_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_final_kept_total=' +
              str(final_kept_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_final_mean_kept_per_chunk=%.6f' %
              float(final_mean))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_tail_candidates_total=' +
              str(tail_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_replaced_total=' +
              str(replaced_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_skip_similar_total=' +
              str(skip_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_mean_replace_per_chunk=%.6f' %
              float(mean_replace))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_similarity_sum=%.6f' %
              float(similarity_sum))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_similarity_count=' +
              str(similarity_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_similarity_mean=%.6f' %
              float(similarity_mean))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_similarity_max=%.6f' %
              float(similarity_max))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_rel_gap_sum=%.6f' %
              float(rel_gap_sum))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_rel_gap_count=' +
              str(rel_gap_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_rel_gap_mean=%.6f' %
              float(rel_gap_mean))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_permanent_rejected_total=0')
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_current_task_chunks=' +
              str(current_task_chunks))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_historical_task_chunks=' +
              str(historical_task_chunks))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_current_anchor_replace_count=' +
              str(current_anchor_replace_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_historical_anchor_replace_count=' +
              str(historical_anchor_replace_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_historical_top4_count=' +
              str(historical_top4_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] gss_anchor_replace_historical_fallback_count=' +
              str(historical_fallback_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] current_task_chunks=' + str(current_task_chunks))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] historical_task_chunks=' + str(historical_task_chunks))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] current_anchor_replace_count=' +
              str(current_anchor_replace_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] historical_anchor_replace_count=' +
              str(historical_anchor_replace_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] historical_top4_count=' +
              str(historical_top4_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] historical_fallback_count=' +
              str(historical_fallback_count))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] final_mean_kept_per_chunk=%.6f' %
              float(final_mean))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] replaced_total=' + str(replaced_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] skip_similar_total=' + str(skip_total))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] mean_replace_per_chunk=%.6f' %
              float(mean_replace))
        print('[GSS-ANCHOR-REPLACE-SUMMARY] permanent_rejected_total=0')
        if coreset_selection_functions.gss_anchor_replace_uses_prob_replace(self.selection_strategy):
            self.print_gss_anchor_replace_prob_summary()

    def print_gss_anchor_replace_prob_summary(self):
        stats = self.gss_anchor_replace_stats
        candidate_total = int(stats.get('gss_anchor_replace_prob_candidate_total', 0))
        accept_total = int(stats.get('gss_anchor_replace_prob_accept_total', 0))
        reject_total = int(stats.get('gss_anchor_replace_prob_reject_total', 0))
        p_sum = float(stats.get('gss_anchor_replace_prob_p_sum', 0.0))
        p_count = int(stats.get('gss_anchor_replace_prob_p_count', 0))
        p_min = stats.get('gss_anchor_replace_prob_p_min', None)
        p_max = stats.get('gss_anchor_replace_prob_p_max', None)
        target_sim_sum = float(stats.get('gss_anchor_replace_prob_target_sim_sum', 0.0))
        target_sim_count = int(stats.get('gss_anchor_replace_prob_target_sim_count', 0))
        candidate_sim_sum = float(stats.get('gss_anchor_replace_prob_candidate_sim_sum', 0.0))
        candidate_sim_count = int(stats.get('gss_anchor_replace_prob_candidate_sim_count', 0))
        u_sum = float(stats.get('gss_anchor_replace_prob_u_sum', 0.0))
        u_count = int(stats.get('gss_anchor_replace_prob_u_count', 0))
        accept_rate = accept_total / max(candidate_total, 1)
        p_mean = p_sum / max(p_count, 1)
        target_sim_mean = target_sim_sum / max(target_sim_count, 1)
        candidate_sim_mean = candidate_sim_sum / max(candidate_sim_count, 1)
        u_mean = u_sum / max(u_count, 1)
        p_min = 0.0 if p_min is None else float(p_min)
        p_max = 0.0 if p_max is None else float(p_max)
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_enabled=True')
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_conservativeness=%.6f' %
              float(self.gss_anchor_replace_prob_conservativeness))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_candidate_total=' +
              str(candidate_total))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_accept_total=' +
              str(accept_total))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_reject_total=' +
              str(reject_total))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_accept_rate=%.6f' %
              float(accept_rate))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_p_sum=%.6f' %
              float(p_sum))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_p_count=' +
              str(p_count))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_p_mean=%.6f' %
              float(p_mean))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_p_min=%.6f' %
              float(p_min))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_p_max=%.6f' %
              float(p_max))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_target_sim_sum=%.6f' %
              float(target_sim_sum))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_target_sim_count=' +
              str(target_sim_count))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_target_sim_mean=%.6f' %
              float(target_sim_mean))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_candidate_sim_sum=%.6f' %
              float(candidate_sim_sum))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_candidate_sim_count=' +
              str(candidate_sim_count))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_candidate_sim_mean=%.6f' %
              float(candidate_sim_mean))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_u_sum=%.6f' %
              float(u_sum))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_u_count=' +
              str(u_count))
        print('[GSS-ANCHOR-REPLACE-PROB-SUMMARY] gss_anchor_replace_prob_u_mean=%.6f' %
              float(u_mean))

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
