# -*-coding:utf8-*-

import torch

from coreset_selection import coreset_selection_functions as funcs


def _ranked_items():
    return [(idx, 1.0 - 0.1 * idx) for idx in range(8)]


def _one_hot(pos, dim=16):
    vec = torch.zeros(dim, dtype=torch.float32)
    vec[pos] = 1.0
    return vec


def _selected_ids(selected_items):
    return [int(item[0]) for item in selected_items]


def _base_anchor_grads():
    return {
        0: _one_hot(0),
        1: _one_hot(1),
        2: _one_hot(2),
        3: _one_hot(3),
    }


def case1_top4_anchors_enter_selected():
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(0), 5: _one_hot(1), 6: _one_hot(2), 7: _one_hot(3)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90
    )
    assert _selected_ids(selected) == [0, 1, 2, 3]


def case2_tail_similar_skips_replacement():
    stats = funcs.make_gss_anchor_replace_stats()
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(0), 5: _one_hot(1), 6: _one_hot(2), 7: _one_hot(3)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats
    )
    assert _selected_ids(selected) == [0, 1, 2, 3]
    assert stats['gss_anchor_replace_skip_similar_total'] == 4
    assert stats['gss_anchor_replace_replaced_total'] == 0


def case3_low_similarity_replaces_lowest_rel_anchor():
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(4), 5: _one_hot(0), 6: _one_hot(1), 7: _one_hot(2)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90
    )
    ids = _selected_ids(selected)
    assert len(ids) == 4
    assert 4 in ids
    assert 3 not in ids


def case4_tail_candidates_use_updated_selected():
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(4), 5: _one_hot(4), 6: _one_hot(0), 7: _one_hot(1)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90
    )
    ids = _selected_ids(selected)
    assert 4 in ids
    assert 5 not in ids

    grads = _base_anchor_grads()
    grads.update({4: _one_hot(4), 5: _one_hot(5), 6: _one_hot(0), 7: _one_hot(1)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90
    )
    ids = _selected_ids(selected)
    assert 5 in ids
    assert 4 not in ids
    assert 3 not in ids


def case5_all_tasks_strategy_disables_historical_fallback():
    assert not funcs.gss_anchor_replace_uses_anchor('rel_gss_anchor_replace', False)
    assert funcs.gss_anchor_replace_uses_anchor('rel_gss_anchor_replace', True)
    assert funcs.gss_anchor_replace_uses_anchor('rel_gss_anchor_replace_all_tasks', False)
    assert funcs.gss_anchor_replace_uses_anchor('rel_gss_anchor_replace_all_tasks', True)
    assert not funcs.gss_anchor_replace_uses_anchor('rel_gss_anchor_replace_hist_top4', False)
    assert funcs.gss_anchor_replace_uses_anchor('rel_gss_anchor_replace_hist_top4', True)
    assert funcs.gss_anchor_replace_uses_historical_rel_top4(
        'rel_gss_anchor_replace_hist_top4', False)
    assert funcs.gss_anchor_replace_resolve_chunk_size(
        selection_strategy='rel_gss_anchor_replace',
        is_current_task=False,
        selection_chunk_size=4,
        base_incremental_size=4,
        remaining_select_size=4
    ) == 1
    assert funcs.gss_anchor_replace_resolve_chunk_size(
        selection_strategy='rel_gss_anchor_replace_all_tasks',
        is_current_task=False,
        selection_chunk_size=4,
        base_incremental_size=4,
        remaining_select_size=4
    ) == 4
    assert funcs.gss_anchor_replace_resolve_chunk_size(
        selection_strategy='rel_gss_anchor_replace',
        is_current_task=True,
        selection_chunk_size=4,
        base_incremental_size=4,
        remaining_select_size=4
    ) == 4
    assert funcs.gss_anchor_replace_resolve_chunk_size(
        selection_strategy='rel_gss_anchor_replace_hist_top4',
        is_current_task=False,
        selection_chunk_size=4,
        base_incremental_size=4,
        remaining_select_size=4
    ) == 4


def case6_final_kept_per_chunk_stays_four():
    stats = funcs.make_gss_anchor_replace_stats()
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(0), 5: _one_hot(1), 6: _one_hot(2), 7: _one_hot(3)})
    for _ in range(3):
        selected = funcs.select_anchor_replace_ranked_items(
            sorted_loss_diffs=_ranked_items(),
            gradients_by_id=grads,
            window_size=8,
            anchor_size=4,
            sim_threshold=0.90,
            stats=stats
        )
        assert len(selected) == 4
    mean_kept = stats['gss_anchor_replace_final_kept_total'] / stats['gss_anchor_replace_chunks_total']
    assert abs(mean_kept - 4.0) < 1e-8


def case7_current_and_historical_stats_record_anchor_replace():
    stats = funcs.make_gss_anchor_replace_stats()
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(4), 5: _one_hot(5), 6: _one_hot(6), 7: _one_hot(7)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=True
    )
    assert len(selected) == 4
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=False
    )
    assert len(selected) == 4
    assert stats['gss_anchor_replace_current_task_chunks'] == 1
    assert stats['gss_anchor_replace_historical_task_chunks'] == 1
    assert stats['gss_anchor_replace_current_anchor_replace_count'] == 1
    assert stats['gss_anchor_replace_historical_anchor_replace_count'] == 1
    assert stats['gss_anchor_replace_historical_fallback_count'] == 0
    assert stats['gss_anchor_replace_final_kept_total'] / stats['gss_anchor_replace_chunks_total'] == 4.0


def case_hist_top4_strategy_uses_historical_rel_top4():
    stats = funcs.make_gss_anchor_replace_stats()
    grads = _base_anchor_grads()
    grads.update({4: _one_hot(4), 5: _one_hot(5), 6: _one_hot(6), 7: _one_hot(7)})
    selected = funcs.select_anchor_replace_ranked_items(
        sorted_loss_diffs=_ranked_items(),
        gradients_by_id=grads,
        window_size=8,
        anchor_size=4,
        sim_threshold=0.90,
        stats=stats,
        is_current_task=True
    )
    assert len(selected) == 4
    funcs.record_gss_anchor_replace_historical_top4(stats, final_kept=4)
    assert stats['gss_anchor_replace_current_anchor_replace_count'] > 0
    assert stats['gss_anchor_replace_historical_anchor_replace_count'] == 0
    assert stats['gss_anchor_replace_historical_fallback_count'] == 0
    assert stats['gss_anchor_replace_historical_top4_count'] > 0
    mean_kept = stats['gss_anchor_replace_final_kept_total'] / stats['gss_anchor_replace_chunks_total']
    assert abs(mean_kept - 4.0) < 1e-8


def main():
    cases = [
        case1_top4_anchors_enter_selected,
        case2_tail_similar_skips_replacement,
        case3_low_similarity_replaces_lowest_rel_anchor,
        case4_tail_candidates_use_updated_selected,
        case5_all_tasks_strategy_disables_historical_fallback,
        case6_final_kept_per_chunk_stays_four,
        case7_current_and_historical_stats_record_anchor_replace,
        case_hist_top4_strategy_uses_historical_rel_top4,
    ]
    for case in cases:
        case()
        print('[synthetic-anchor-replace] PASS ' + case.__name__)
    print('PASS synthetic_anchor_replace_diagnostic')


if __name__ == '__main__':
    main()
