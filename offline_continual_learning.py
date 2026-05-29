# -*-coding:utf8-*-

import argparse
import os
import numpy as np

import utils
from continual_learning import continual_runner
from dataset import idataset


def make_selection_params(opts):
    selection_params = {
        'init_size': 0,
        'class_balance': False,
        'only_new_data': True,
        'cur_train_steps': opts.cur_train_steps,
        'cur_train_lr': opts.cur_train_lr,
        'selection_steps': opts.selection_steps,
        'ideal_logit': True,
        'logit_compute_mode': 'end_task',
        'loss_params': {
            'ce_factor': 1.0,
            'mse_factor': opts.slt_mse_factor
        },
        'selection_strategy': opts.selection_strategy,
        'selection_chunk_size': opts.selection_chunk_size,
        'gss_anchor_replace_window': opts.gss_anchor_replace_window,
        'gss_anchor_replace_anchor_size': opts.gss_anchor_replace_anchor_size,
        'gss_anchor_replace_sim_threshold': opts.gss_anchor_replace_sim_threshold,
        'gss_grad_layer': opts.gss_grad_layer,
        'div_lambda': opts.div_lambda,
        'div_candidate_ratio': opts.div_candidate_ratio,
        'div_feature_source': opts.div_feature_source,
        'div_feature_layer': opts.div_feature_layer,
        'div_verbose': opts.div_verbose,
        'filter_sim_threshold': opts.filter_sim_threshold,
        'filter_feature_source': opts.filter_feature_source,
        'filter_feature_layer': opts.filter_feature_layer,
        'filter_reject_dominated': opts.filter_reject_dominated,
        'filter_verbose': opts.filter_verbose,
        'ref_train_params': {
            'lr': opts.ref_train_lr,
            'epochs': opts.ref_train_epoch,
            'batch_size': 32,
            'eval_batch_size': 20,
            'use_cuda': bool(opts.use_cuda),
            'early_stop': -1,
            'log_steps': 100,
            'opt_type': 'sgd',
            'loss_params': {
                'ce_factor': 1.0,
                'mse_factor': 0.0
            },
            'ref_sample_per_task': opts.ref_sample_per_task  # default 0
        }
    }
    return selection_params


def main(opts):
    if not os.path.exists(opts.local_path):
        os.makedirs(opts.local_path)
    # make data loaders
    model_params, transforms, eval_transforms, task_dic, train_loaders, train_sub_loaders_wo_aug, test_loaders =\
        idataset.get_dataset(
            opts=opts
        )
    # modify selection params
    selection_params = make_selection_params(opts=opts)
    for k in selection_params.keys():
        print(k, '\t\t', selection_params[k])
    # build continual runner
    train_params = {
        'lr': opts.lr,
        'alpha': opts.alpha,
        'epochs': opts.epochs,
        'batch_size': opts.batch_size,
        'mem_batch_size': opts.mem_batch_size,
        'eval_batch_size': 20,
        'use_cuda': bool(opts.use_cuda),
        'early_stop': -1,
        'log_steps': 100,
        'opt_type': opts.opt_type
    }
    if opts.beta > 0:
        train_params['beta'] = opts.beta
    if opts.runner_type == 'coreset':
        runner = continual_runner.ContinualRunner(
            local_path=opts.local_path,
            model_params=model_params,
            transforms=transforms,
            train_params=train_params,
            selection_params=selection_params,
            use_cuda=bool(opts.use_cuda),
            task_dic=task_dic,
            buffer_size=opts.buffer_size,
            seed=opts.seed,
            replay_mode=opts.replay_mode,
            selection_transforms=eval_transforms if bool(opts.slt_wo_aug) else None,
            extra_data_mode=opts.extra_data.split(','),
            buffer_type=opts.buffer_type
        )
    else:
        raise ValueError('Invalid runner type')
    # continual training and update memory
    for i in range(len(task_dic)):
        if opts.runner_type == 'coreset':
            accs = runner.train_single_task(
                train_loader=train_loaders[i],
                eval_loaders=test_loaders,
                verbose=True,
                do_evaluation=True
            )
        else:
            raise ValueError('Invalid runner type')
        print('accuracies on testset after task', i, 'is:', accs, np.mean(accs))
        if opts.runner_type == 'coreset':
            runner.update_buffer(
                full_train_loader=train_sub_loaders_wo_aug[i],
                sub_loader=train_sub_loaders_wo_aug[i],
                next_loader=None
            )
        else:
            raise ValueError('Invalid runner type')
        runner.next_task(dump_buffer=True)


if __name__ == '__main__':
    """
    selection parameters are added in slt_config.py
    """
    parser = argparse.ArgumentParser('offline continual learning')
    parser.add_argument('--local_path', type=str)
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--setting', type=str, default='greedy')
    parser.add_argument('--buffer_size', type=int)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--beta', type=float, default=0)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--mem_batch_size', type=int)
    parser.add_argument('--use_cuda', type=int)
    parser.add_argument('--opt_type', type=str)
    parser.add_argument('--slt_wo_aug', type=int, default=0)
    parser.add_argument('--holdout_set', type=str, default='full')
    parser.add_argument('--replay_mode', type=str, default='full')
    parser.add_argument('--use_bn', type=int, default=0)
    parser.add_argument('--limit_per_task', type=int, default=1000)
    parser.add_argument('--runner_type', type=str, default='coreset')
    parser.add_argument('--update_mode', type=str, default='random')
    parser.add_argument('--extra_data', type=str, default='')
    parser.add_argument('--slt_mse_factor', type=float, default=-1)
    parser.add_argument('--cur_train_steps', type=int, default=-1)
    parser.add_argument('--ref_train_epoch', type=int, default=-1)
    parser.add_argument('--selection_steps', type=int, default=-1)
    parser.add_argument('--ref_train_lr', type=float, default=-1)
    parser.add_argument('--cur_train_lr', type=float, default=-1)
    parser.add_argument('--aug_type', type=str, default='greedy')
    parser.add_argument('--buffer_type', type=str, default='coreset')
    parser.add_argument('--ref_sample_per_task', type=int, default=-1)
    parser.add_argument('--memory_allocation', type=str, default='equal', choices=['equal'])
    parser.add_argument('--selection_strategy', type=str, default='rel',
                        choices=['rel', 'rel_filter', 'rel_diversity', 'rel_gss_anchor_replace',
                                 'rel_gss_anchor_replace_all_tasks'])
    parser.add_argument('--selection_chunk_size', type=int, default=0)
    parser.add_argument('--gss_anchor_replace_window', type=int, default=8)
    parser.add_argument('--gss_anchor_replace_anchor_size', type=int, default=4)
    parser.add_argument('--gss_anchor_replace_sim_threshold', type=float, default=0.90)
    parser.add_argument('--gss_grad_layer', type=str, default='classifier')
    parser.add_argument('--div_lambda', type=float, default=0.1)
    parser.add_argument('--div_candidate_ratio', type=int, default=3)
    parser.add_argument('--div_feature_source', type=str, default='current_model', choices=['current_model', 'holdout_model'])
    parser.add_argument('--div_feature_layer', type=str, default='penultimate')
    parser.add_argument('--div_verbose', action='store_true')
    parser.add_argument('--filter_sim_threshold', type=float, default=0.85)
    parser.add_argument('--filter_feature_source', type=str, default='current_model', choices=['current_model', 'holdout_model'])
    parser.add_argument('--filter_feature_layer', type=str, default='penultimate')
    parser.add_argument('--filter_reject_dominated', action='store_true')
    parser.add_argument('--filter_verbose', action='store_true')
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()

    utils.set_random_seed(seed=args.seed)
    print('script\t\t', 'offline_continual_learning.py')
    print('local path\t\t', args.local_path)
    print('data path\t\t', args.data_path)
    print('dataset\t\t', args.dataset)
    print('setting\t\t', args.setting)
    print('buffer size\t\t', args.buffer_size)
    print('alpha\t\t', args.alpha)
    print('beta\t\t', args.beta)
    print('lr\t\t', args.lr)
    print('epochs\t\t', args.epochs)
    print('batch size\t\t', args.batch_size)
    print('mem batch size\t\t', args.mem_batch_size)
    print('opt type\t\t', args.opt_type)
    print('select without augmentation\t\t', args.slt_wo_aug)
    print('holdout set\t\t', args.holdout_set)
    print('replay mode\t\t', args.replay_mode)
    print('use bn\t\t', args.use_bn)
    print('limit per task\t\t', args.limit_per_task)
    print('runner type\t\t', args.runner_type)
    print('update mode\t\t', args.update_mode)
    print('extra data\t\t', args.extra_data)
    print('slt_mse_factor\t\t', args.slt_mse_factor)
    print('cur_train_steps\t\t', args.cur_train_steps)
    print('ref_train_epoch\t\t', args.ref_train_epoch)
    print('selection steps\t\t', args.selection_steps)
    print('ref train lr\t\t', args.ref_train_lr)
    print('cur train lr\t\t', args.cur_train_lr)
    print('aug type\t\t', args.aug_type)
    print('buffer type\t\t', args.buffer_type)
    print('ref sample per task\t\t', args.ref_sample_per_task)
    print('memory allocation\t\t', args.memory_allocation)
    print('selection strategy\t\t', args.selection_strategy)
    print('selection chunk size\t\t', args.selection_chunk_size)
    print('gss anchor replace window\t\t', args.gss_anchor_replace_window)
    print('gss anchor replace anchor size\t\t', args.gss_anchor_replace_anchor_size)
    print('gss anchor replace sim threshold\t\t', args.gss_anchor_replace_sim_threshold)
    print('gss grad layer\t\t', args.gss_grad_layer)
    print('div lambda\t\t', args.div_lambda)
    print('div candidate ratio\t\t', args.div_candidate_ratio)
    print('div feature source\t\t', args.div_feature_source)
    print('div feature layer\t\t', args.div_feature_layer)
    print('div verbose\t\t', args.div_verbose)
    print('filter sim threshold\t\t', args.filter_sim_threshold)
    print('filter feature source\t\t', args.filter_feature_source)
    print('filter feature layer\t\t', args.filter_feature_layer)
    print('filter reject dominated\t\t', args.filter_reject_dominated)
    print('filter verbose\t\t', args.filter_verbose)
    print('seed\t\t', args.seed)
    main(opts=args)
