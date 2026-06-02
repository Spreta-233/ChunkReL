# -*-coding:utf8-*-

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


def _prob_base_items():
    return [
        (1, 0.90),
        (2, 0.85),
        (3, 0.82),
        (4, 0.80),
        (5, 0.59),
    ]


def _prob_base_grads():
    return {
        1: _one_hot(0),
        2: _one_hot(1),
        3: _one_hot(2),
        4: _one_hot(0),
        5: _one_hot(3),
    }


def TEST_STRATEGY_RECOGNITION():
    strategy = 'rel_gss_anchor_replace_hist_top4_prob'
    assert funcs.gss_anchor_replace_is_strategy(strategy)
    assert funcs.gss_anchor_replace_uses_anchor(strategy, True)
    assert not funcs.gss_anchor_replace_uses_anchor(strategy, False)
    assert funcs.gss_anchor_replace_uses_historical_rel_top4(strategy, False)
    assert funcs.gss_anchor_replace_uses_prob_replace(strategy)
    print('TEST_STRATEGY_RECOGNITION PASS')


def TEST_HISTORY_REL_TOP4_ONLY():
    strategy = 'rel_gss_anchor_replace_hist_top4_prob'
    sorted_loss_diffs = [(idx, 1.0 - idx * 0.01) for idx in range(1, 9)]
    selected = list(sorted_loss_diffs[:4])
    stats = funcs.make_gss_anchor_replace_stats()
    funcs.record_gss_anchor_replace_historical_top4(stats, len(selected))
    assert _selected_ids(selected) == [1, 2, 3, 4]
    assert funcs.gss_anchor_replace_uses_historical_rel_top4(strategy, False)
    assert not funcs.gss_anchor_replace_uses_anchor(strategy, False)
    assert stats['gss_anchor_replace_historical_top4_count'] == 1
    assert stats['gss_anchor_replace_replaced_total'] == 0
    assert stats['gss_anchor_replace_permanent_rejected_total'] == 0
    print('TEST_HISTORY_REL_TOP4_ONLY PASS')


def TEST_SKIP_SIMILAR_DOES_NOT_ENTER_PROB():
    sorted_loss_diffs = _prob_base_items()
    grads = _prob_base_grads()
    grads[5] = _one_hot(0)
    stats = funcs.make_gss_anchor_replace_stats()
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=5,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=True,
        probabilistic_replace=True,
        prob_rng=FixedRng([0.0])
    )
    assert _selected_ids(selected) == [1, 2, 3, 4]
    assert stats['gss_anchor_replace_skip_similar_total'] == 1
    assert stats['gss_anchor_replace_prob_candidate_total'] == 0
    assert stats['gss_anchor_replace_prob_u_count'] == 0
    assert stats['gss_anchor_replace_replaced_total'] == 0
    print('TEST_SKIP_SIMILAR_DOES_NOT_ENTER_PROB PASS')


def TEST_PROB_ACCEPTS_ELIGIBLE_CANDIDATE():
    stats = funcs.make_gss_anchor_replace_stats()
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_prob_base_items(),
        gradients_by_id=_prob_base_grads(),
        window_size=5,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=True,
        probabilistic_replace=True,
        prob_rng=FixedRng([0.10])
    )
    assert _selected_ids(selected) == [1, 2, 3, 5]
    assert stats['gss_anchor_replace_prob_candidate_total'] == 1
    assert stats['gss_anchor_replace_prob_accept_total'] == 1
    assert stats['gss_anchor_replace_prob_reject_total'] == 0
    assert stats['gss_anchor_replace_replaced_total'] == 1
    assert stats['gss_anchor_replace_permanent_rejected_total'] == 0
    assert abs(stats['gss_anchor_replace_prob_target_sim_sum'] - 1.0) < 1e-6
    assert abs(stats['gss_anchor_replace_prob_candidate_sim_sum']) < 1e-6
    print('TEST_PROB_ACCEPTS_ELIGIBLE_CANDIDATE PASS')


def TEST_OLD_DIRECT_REPLACE_UNCHANGED():
    stats = funcs.make_gss_anchor_replace_stats()
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_prob_base_items(),
        gradients_by_id=_prob_base_grads(),
        window_size=5,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=True,
        probabilistic_replace=False
    )
    assert _selected_ids(selected) == [1, 2, 3, 5]
    assert stats['gss_anchor_replace_replaced_total'] == 1
    assert stats['gss_anchor_replace_prob_candidate_total'] == 0
    assert stats['gss_anchor_replace_permanent_rejected_total'] == 0
    print('TEST_OLD_DIRECT_REPLACE_UNCHANGED PASS')


def TEST_PROB_REJECTS_ELIGIBLE_CANDIDATE():
    stats = funcs.make_gss_anchor_replace_stats()
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_prob_base_items(),
        gradients_by_id=_prob_base_grads(),
        window_size=5,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=True,
        probabilistic_replace=True,
        prob_rng=FixedRng([0.99])
    )
    assert _selected_ids(selected) == [1, 2, 3, 4]
    assert stats['gss_anchor_replace_prob_candidate_total'] == 1
    assert stats['gss_anchor_replace_prob_accept_total'] == 0
    assert stats['gss_anchor_replace_prob_reject_total'] == 1
    assert stats['gss_anchor_replace_replaced_total'] == 0
    assert stats['gss_anchor_replace_permanent_rejected_total'] == 0
    print('TEST_PROB_REJECTS_ELIGIBLE_CANDIDATE PASS')


def TEST_SELECTION_SIZE_STAYS_AT_K():
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
        prob_rng=FixedRng([0.1, 0.1, 0.1, 0.1])
    )
    assert len(selected) == 4
    assert stats['gss_anchor_replace_final_kept_total'] == 4
    print('TEST_SELECTION_SIZE_STAYS_AT_K PASS')


def TEST_DETERMINISTIC_SEEDED_RNG():
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
            prob_rng=np.random.RandomState(17)
        )
        return _selected_ids(selected), stats

    ids_1, stats_1 = run_once()
    ids_2, stats_2 = run_once()
    assert ids_1 == ids_2
    assert stats_1['gss_anchor_replace_prob_accept_total'] == \
        stats_2['gss_anchor_replace_prob_accept_total']
    assert abs(stats_1['gss_anchor_replace_prob_u_sum'] -
               stats_2['gss_anchor_replace_prob_u_sum']) < 1e-12
    print('TEST_DETERMINISTIC_SEEDED_RNG PASS')


def main():
    TEST_STRATEGY_RECOGNITION()
    TEST_HISTORY_REL_TOP4_ONLY()
    TEST_SKIP_SIMILAR_DOES_NOT_ENTER_PROB()
    TEST_PROB_ACCEPTS_ELIGIBLE_CANDIDATE()
    TEST_OLD_DIRECT_REPLACE_UNCHANGED()
    TEST_PROB_REJECTS_ELIGIBLE_CANDIDATE()
    TEST_SELECTION_SIZE_STAYS_AT_K()
    TEST_DETERMINISTIC_SEEDED_RNG()
    print('PASS synthetic_gss_anchor_prob_diagnostic')


if __name__ == '__main__':
    main()
