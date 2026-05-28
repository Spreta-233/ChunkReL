import torch

from coreset_selection.coreset_selection_functions import apply_ccrf_chunk_filter, apply_gss_temp_chunk_filter
from coreset_selection.selection_agent import resolve_ccrf_effective_chunk_size


def run_case(name, labels, features, expected_kept, expected_rejected, threshold=0.85,
             is_current_task=True, chunk_size=4):
    candidate_ids = list(range(len(labels)))
    effective_chunk_size = resolve_ccrf_effective_chunk_size(
        selection_chunk_size=chunk_size,
        base_incremental_size=1,
        remaining_select_size=len(labels),
        is_current_task=is_current_task
    )
    candidate_ids = candidate_ids[:effective_chunk_size]
    labels = labels[:effective_chunk_size]
    features = features[:effective_chunk_size]
    kept, rejected, stats = apply_ccrf_chunk_filter(
        candidate_ids=candidate_ids,
        candidate_labels=labels,
        features=torch.tensor(features, dtype=torch.float32),
        sim_threshold=threshold,
        is_current_task=is_current_task
    )
    ok = kept == expected_kept and rejected == expected_rejected
    print(
        name + ': kept=' + str(kept) +
        ' rejected=' + str(rejected) +
        ' expected_kept=' + str(expected_kept) +
        ' expected_rejected=' + str(expected_rejected) +
        ' ok=' + str(ok)
    )
    if not ok:
        raise AssertionError(name + ' failed with stats=' + str(stats))
    return stats


def run_gss_case(name, candidate_ids, gradients, expected_kept, expected_deferred,
                 is_current_task=True, chunk_size=4, threshold=0.95):
    effective_chunk_size = resolve_ccrf_effective_chunk_size(
        selection_chunk_size=chunk_size,
        base_incremental_size=1,
        remaining_select_size=len(candidate_ids),
        is_current_task=is_current_task
    )
    candidate_ids = candidate_ids[:effective_chunk_size]
    gradients = gradients[:effective_chunk_size]
    kept, deferred, permanent_rejected, stats = apply_gss_temp_chunk_filter(
        candidate_ids=candidate_ids,
        gradients=torch.tensor(gradients, dtype=torch.float32),
        sim_threshold=threshold,
        is_current_task=is_current_task
    )
    ok = kept == expected_kept and deferred == expected_deferred and permanent_rejected == []
    print(
        name + ': kept=' + str(kept) +
        ' deferred=' + str(deferred) +
        ' permanent_rejected=' + str(permanent_rejected) +
        ' ok=' + str(ok)
    )
    if not ok:
        raise AssertionError(name + ' failed with stats=' + str(stats))
    return stats


def main():
    case1_stats = run_case(
        name='case1_same_class_high_sim_only_first_pair',
        labels=[0, 0, 1, 1],
        features=[
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        expected_kept=[0, 2, 3],
        expected_rejected=[1]
    )
    case2_stats = run_case(
        name='case2_cross_class_high_sim_ignored',
        labels=[0, 1, 2, 3],
        features=[
            [1.0, 0.00, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
        ],
        expected_kept=[0, 1, 2, 3],
        expected_rejected=[]
    )
    case3_stats = run_case(
        name='case3_all_same_class_high_sim',
        labels=[0, 0, 0, 0],
        features=[
            [1.0, 0.00, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
        ],
        expected_kept=[0],
        expected_rejected=[1, 2, 3]
    )
    case4_stats = run_case(
        name='case4_current_task_ccrf_filtering',
        labels=[0, 0, 1, 1],
        features=[
            [1.0, 0.0, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        expected_kept=[0, 2, 3],
        expected_rejected=[1],
        is_current_task=True,
        chunk_size=4
    )
    case5_stats = run_case(
        name='case5_historical_task_top1_fallback',
        labels=[0, 0, 0, 0],
        features=[
            [1.0, 0.00, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
        ],
        expected_kept=[0],
        expected_rejected=[],
        is_current_task=False,
        chunk_size=4
    )
    if case5_stats['same_class_pair_checked_count'] != 0 or case5_stats['rejected_count'] != 0:
        raise AssertionError('case5 should not enter similarity filtering')
    case6_stats = run_gss_case(
        name='case6_gss_temp_current_task_defers_duplicate_gradient',
        candidate_ids=[0, 1, 2, 3],
        gradients=[
            [1.0, 0.00, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.0, 1.00, 0.0, 0.0],
            [0.0, 0.00, 1.0, 0.0],
        ],
        expected_kept=[0, 2, 3],
        expected_deferred=[1],
        is_current_task=True
    )
    case7_round1_stats = run_gss_case(
        name='case7_round1_deferred_not_permanently_rejected',
        candidate_ids=[0, 1, 2, 3],
        gradients=[
            [1.0, 0.00, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.0, 1.00, 0.0, 0.0],
            [0.0, 0.00, 1.0, 0.0],
        ],
        expected_kept=[0, 2, 3],
        expected_deferred=[1],
        is_current_task=True
    )
    case7_round2_stats = run_gss_case(
        name='case7_round2_deferred_sample_can_return',
        candidate_ids=[1, 4, 5, 6],
        gradients=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        expected_kept=[1, 4, 5, 6],
        expected_deferred=[],
        is_current_task=True
    )
    case8_stats = run_gss_case(
        name='case8_gss_temp_historical_top1_fallback',
        candidate_ids=[0, 1, 2, 3],
        gradients=[
            [1.0, 0.00, 0.0, 0.0],
            [0.99, 0.01, 0.0, 0.0],
            [0.98, 0.02, 0.0, 0.0],
            [0.97, 0.03, 0.0, 0.0],
        ],
        expected_kept=[0],
        expected_deferred=[],
        is_current_task=False
    )
    if case8_stats['grad_pair_checked_count'] != 0:
        raise AssertionError('case8 should not enter gradient filtering')
    print('case1_stats=' + str(case1_stats))
    print('case2_stats=' + str(case2_stats))
    print('case3_stats=' + str(case3_stats))
    print('case4_stats=' + str(case4_stats))
    print('case5_stats=' + str(case5_stats))
    print('case6_stats=' + str(case6_stats))
    print('case7_round1_stats=' + str(case7_round1_stats))
    print('case7_round2_stats=' + str(case7_round2_stats))
    print('case8_stats=' + str(case8_stats))
    print('synthetic_ccrf_diagnostic: PASS')


if __name__ == '__main__':
    main()
