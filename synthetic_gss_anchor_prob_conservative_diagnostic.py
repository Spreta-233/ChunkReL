# -*-coding:utf8-*-

import subprocess
import sys

import numpy as np
import torch

from coreset_selection import coreset_selection_functions as funcs


class FixedRng(object):
    def __init__(self, values):
        self.values = list(values)
        self.pos = 0

    def random_sample(self):
        if self.pos >= len(self.values):
            return float(self.values[-1])
        value = float(self.values[self.pos])
        self.pos += 1
        return value


def _one_hot(pos, dim=16):
    vec = torch.zeros(dim, dtype=torch.float32)
    vec[pos] = 1.0
    return vec


def _selected_ids(selected_items):
    return [int(item[0]) for item in selected_items]


def _base_items():
    return [
        (1, 0.90),
        (2, 0.85),
        (3, 0.82),
        (4, 0.80),
        (5, 0.59),
    ]


def _base_grads():
    return {
        1: _one_hot(0),
        2: _one_hot(1),
        3: _one_hot(2),
        4: _one_hot(0),
        5: _one_hot(3),
    }


def TEST_ARGPARSE_HAS_CONSERVATIVENESS():
    output = subprocess.check_output(
        [sys.executable, 'offline_continual_learning.py', '--help'],
        stderr=subprocess.STDOUT
    ).decode('utf-8', errors='ignore')
    assert '--gss_anchor_replace_prob_conservativeness' in output
    print('TEST_ARGPARSE_HAS_CONSERVATIVENESS PASS')


def TEST_LAMBDA_ONE_MATCHES_OLD_FORMULA():
    target_sim = 1.0
    candidate_sim = 0.0
    old_p = (target_sim + 1.0) / ((target_sim + 1.0) + (candidate_sim + 1.0))
    new_p = funcs.compute_gss_anchor_replace_prob(
        target_sim=target_sim,
        candidate_sim=candidate_sim,
        conservativeness=1.0
    )
    assert abs(old_p - new_p) < 1e-12
    print('TEST_LAMBDA_ONE_MATCHES_OLD_FORMULA PASS')


def TEST_CONSERVATIVENESS_MONOTONIC():
    p1 = funcs.compute_gss_anchor_replace_prob(1.0, 0.0, conservativeness=1.0)
    p2 = funcs.compute_gss_anchor_replace_prob(1.0, 0.0, conservativeness=2.0)
    p3 = funcs.compute_gss_anchor_replace_prob(1.0, 0.0, conservativeness=3.0)
    assert p1 > p2 > p3
    print('TEST_CONSERVATIVENESS_MONOTONIC PASS')


def TEST_CONSERVATIVENESS_ACCEPT_REJECT_SPLIT():
    stats_lo = funcs.make_gss_anchor_replace_stats()
    selected_lo = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_base_items(),
        gradients_by_id=_base_grads(),
        window_size=5,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats_lo,
        is_current_task=True,
        probabilistic_replace=True,
        prob_rng=FixedRng([0.60]),
        prob_conservativeness=1.0
    )
    stats_hi = funcs.make_gss_anchor_replace_stats()
    selected_hi = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_base_items(),
        gradients_by_id=_base_grads(),
        window_size=5,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats_hi,
        is_current_task=True,
        probabilistic_replace=True,
        prob_rng=FixedRng([0.60]),
        prob_conservativeness=2.0
    )
    assert _selected_ids(selected_lo) == [1, 2, 3, 5]
    assert _selected_ids(selected_hi) == [1, 2, 3, 4]
    assert stats_lo['gss_anchor_replace_prob_accept_total'] == 1
    assert stats_hi['gss_anchor_replace_prob_reject_total'] == 1
    assert stats_hi['gss_anchor_replace_prob_conservativeness'] == 2.0
    print('TEST_CONSERVATIVENESS_ACCEPT_REJECT_SPLIT PASS')


def TEST_INVALID_CONSERVATIVENESS_REJECTED():
    try:
        funcs.compute_gss_anchor_replace_prob(1.0, 0.0, conservativeness=0.0)
    except ValueError:
        print('TEST_INVALID_CONSERVATIVENESS_REJECTED PASS')
        return
    raise AssertionError('expected ValueError for conservativeness <= 0')


def TEST_HISTORY_REL_TOP4_ONLY():
    strategy = 'rel_gss_anchor_replace_hist_top4_prob'
    sorted_loss_diffs = [(idx, 1.0 - idx * 0.01) for idx in range(1, 9)]
    selected = list(sorted_loss_diffs[:4])
    stats = funcs.make_gss_anchor_replace_stats()
    funcs.record_gss_anchor_replace_historical_top4(stats, len(selected))
    assert _selected_ids(selected) == [1, 2, 3, 4]
    assert funcs.gss_anchor_replace_uses_historical_rel_top4(strategy, False)
    assert not funcs.gss_anchor_replace_uses_anchor(strategy, False)
    assert stats['gss_anchor_replace_replaced_total'] == 0
    assert stats['gss_anchor_replace_permanent_rejected_total'] == 0
    print('TEST_HISTORY_REL_TOP4_ONLY PASS')


def TEST_SELECTION_SIZE_AND_SEEDED_DETERMINISM():
    sorted_loss_diffs = [(idx, 1.0 - idx * 0.01) for idx in range(1, 9)]
    grads = {
        1: _one_hot(0),
        2: _one_hot(1),
        3: _one_hot(2),
        4: _one_hot(0),
        5: _one_hot(3),
        6: _one_hot(4),
        7: _one_hot(5),
        8: _one_hot(6),
    }

    def run_once():
        stats = funcs.make_gss_anchor_replace_stats()
        selected = funcs.select_anchor_replace_ranked_items(
            sorted_loss_diffs=sorted_loss_diffs,
            gradients_by_id=grads,
            window_size=8,
            anchor_size=4,
            sim_threshold=0.90,
            stats=stats,
            is_current_task=True,
            probabilistic_replace=True,
            prob_rng=np.random.RandomState(23),
            prob_conservativeness=1.5
        )
        return _selected_ids(selected), stats

    ids_1, stats_1 = run_once()
    ids_2, stats_2 = run_once()
    assert len(ids_1) == 4
    assert ids_1 == ids_2
    assert stats_1['gss_anchor_replace_permanent_rejected_total'] == 0
    assert abs(stats_1['gss_anchor_replace_prob_u_sum'] -
               stats_2['gss_anchor_replace_prob_u_sum']) < 1e-12
    print('TEST_SELECTION_SIZE_AND_SEEDED_DETERMINISM PASS')


def main():
    TEST_ARGPARSE_HAS_CONSERVATIVENESS()
    TEST_LAMBDA_ONE_MATCHES_OLD_FORMULA()
    TEST_CONSERVATIVENESS_MONOTONIC()
    TEST_CONSERVATIVENESS_ACCEPT_REJECT_SPLIT()
    TEST_INVALID_CONSERVATIVENESS_REJECTED()
    TEST_HISTORY_REL_TOP4_ONLY()
    TEST_SELECTION_SIZE_AND_SEEDED_DETERMINISM()
    print('PASS synthetic_gss_anchor_prob_conservative_diagnostic')


if __name__ == '__main__':
    main()
