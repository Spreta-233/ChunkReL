# -*-coding:utf8-*-

import torch

from coreset_selection import coreset_selection_functions as funcs


def _one_hot(pos, dim=16):
    vec = torch.zeros(dim, dtype=torch.float32)
    vec[pos] = 1.0
    return vec


def _selected_ids(selected_items):
    return [int(item[0]) for item in selected_items]


def TEST_GSS_IQP_LOCAL_COMBINATION():
    sorted_loss_diffs = [
        (1, 0.90),
        (2, 0.85),
        (3, 0.82),
        (4, 0.80),
        (5, 0.59),
        (6, 0.58),
        (7, 0.57),
        (8, 0.56),
    ]
    grads = {
        1: _one_hot(0),
        2: _one_hot(0),
        3: _one_hot(0),
        4: _one_hot(0),
        5: _one_hot(1),
        6: _one_hot(2),
        7: _one_hot(3),
        8: _one_hot(4),
    }
    stats = funcs.make_gss_iqp_stats()
    selected = funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=8,
        select_size=4,
        stats=stats,
        is_current_task=True
    )
    assert _selected_ids(selected) == [1, 5, 6, 7]
    assert _selected_ids(selected) != [1, 2, 3, 4]
    assert stats['gss_iqp_current_iqp_chunks'] == 1
    assert stats['gss_iqp_changed_chunks'] == 1
    assert stats['gss_iqp_pairwise_cos_iqp_selected_sum'] < stats['gss_iqp_pairwise_cos_rel_top4_sum']
    print('TEST_GSS_IQP_LOCAL_COMBINATION PASS')


def TEST_GSS_IQP_HISTORY_REL_TOP4():
    sorted_loss_diffs = [(idx, 1.0 - idx * 0.01) for idx in range(1, 9)]
    grads = {idx: _one_hot(idx) for idx, _ in sorted_loss_diffs}
    stats = funcs.make_gss_iqp_stats()
    selected = funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=8,
        select_size=4,
        stats=stats,
        is_current_task=False
    )
    assert _selected_ids(selected) == [1, 2, 3, 4]
    assert stats['gss_iqp_historical_iqp_count'] == 0
    assert stats['gss_iqp_historical_top4_count'] > 0
    print('TEST_GSS_IQP_HISTORY_REL_TOP4 PASS')


def TEST_GSS_IQP_NO_PERMANENT_REJECTION():
    stats = funcs.make_gss_iqp_stats()
    sorted_loss_diffs = [(idx, 1.0 - idx * 0.01) for idx in range(1, 9)]
    grads = {idx: _one_hot(idx) for idx, _ in sorted_loss_diffs}
    funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=8,
        select_size=4,
        stats=stats,
        is_current_task=True
    )
    assert stats['gss_iqp_permanent_rejected_total'] == 0
    print('TEST_GSS_IQP_NO_PERMANENT_REJECTION PASS')


def TEST_GSS_IQP_SELECTION_SIZE():
    stats = funcs.make_gss_iqp_stats()
    sorted_loss_diffs = [(1, 0.90), (2, 0.80), (3, 0.70), (4, 0.60), (5, 0.50)]
    grads = {idx: _one_hot(idx) for idx, _ in sorted_loss_diffs}
    selected = funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=5,
        select_size=4,
        stats=stats,
        is_current_task=True
    )
    assert len(selected) == 4
    short_selected = funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs[:2],
        gradients_by_id=grads,
        window_size=5,
        select_size=4,
        stats=stats,
        is_current_task=True
    )
    assert len(short_selected) == 2
    print('TEST_GSS_IQP_SELECTION_SIZE PASS')


def TEST_GSS_IQP_TIEBREAK():
    sorted_loss_diffs = [(1, 0.90), (2, 0.80), (3, 0.70), (4, 0.60)]
    grads = {idx: _one_hot(idx) for idx, _ in sorted_loss_diffs}
    selected = funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=4,
        select_size=2,
        stats=funcs.make_gss_iqp_stats(),
        is_current_task=True
    )
    assert _selected_ids(selected) == [1, 2]

    sorted_loss_diffs = [(1, 0.90), (2, 0.80), (3, 0.70), (4, 0.65), (5, 0.60)]
    grads = {
        1: torch.tensor([1.0, 0.0, 0.0, 0.0]),
        2: torch.tensor([1.0, 1.0, 1.0, 0.0]),
        3: torch.tensor([1.0, 1.0, -2.0, 0.0]),
        4: torch.tensor([1.0, 1.0, 0.0, 0.0]),
        5: torch.tensor([0.0, 1.0, 0.0, 0.0]),
    }
    selected = funcs.select_gss_iqp_ranked_items(
        sorted_loss_diffs=sorted_loss_diffs,
        gradients_by_id=grads,
        window_size=5,
        select_size=2,
        stats=funcs.make_gss_iqp_stats(),
        is_current_task=True
    )
    assert _selected_ids(selected) == [2, 3]
    print('TEST_GSS_IQP_TIEBREAK PASS')


def main():
    TEST_GSS_IQP_LOCAL_COMBINATION()
    TEST_GSS_IQP_HISTORY_REL_TOP4()
    TEST_GSS_IQP_NO_PERMANENT_REJECTION()
    TEST_GSS_IQP_SELECTION_SIZE()
    TEST_GSS_IQP_TIEBREAK()
    print('PASS synthetic_gss_iqp_diagnostic')


if __name__ == '__main__':
    main()
