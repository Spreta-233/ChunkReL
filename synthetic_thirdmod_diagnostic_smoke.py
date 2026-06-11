# -*-coding:utf8-*-

import contextlib
import copy
import io
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from continual_learning.coreset_buffer import CoresetBuffer
from continual_learning.thirdmod_diagnostic import read_buffer_counts, run_thirdmod_diagnostic


def _numpy_state_equal(left, right):
    return (
        left[0] == right[0]
        and np.array_equal(left[1], right[1])
        and left[2:] == right[2:]
    )


def _capture_rng():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().clone(),
        'cuda': None
    }
    if torch.cuda.is_available():
        state['cuda'] = [item.clone() for item in torch.cuda.get_rng_state_all()]
    return state


def _assert_rng_equal(left, right):
    assert left['python'] == right['python']
    assert _numpy_state_equal(left['numpy'], right['numpy'])
    assert torch.equal(left['torch'], right['torch'])
    if left['cuda'] is not None:
        assert len(left['cuda']) == len(right['cuda'])
        assert all(torch.equal(a, b) for a, b in zip(left['cuda'], right['cuda']))


def _assert_buffer_equal(left, right):
    assert len(left) == len(right)
    for left_task, right_task in zip(left, right):
        assert len(left_task) == len(right_task)
        for left_entry, right_entry in zip(left_task, right_task):
            assert len(left_entry) == len(right_entry)
            for left_value, right_value in zip(left_entry, right_entry):
                if isinstance(left_value, torch.Tensor):
                    assert torch.equal(left_value, right_value)
                else:
                    assert left_value == right_value


def _make_buffer():
    buffer = object.__new__(CoresetBuffer)
    buffer.data = [
        [
            (0, torch.tensor([1.0, 0.0]), 0),
            (1, torch.tensor([0.0, 1.0]), 1)
        ],
        [
            (2, torch.tensor([-1.0, 0.0]), 2),
            (3, torch.tensor([0.0, -1.0]), 3)
        ]
    ]
    return buffer


def _make_model():
    model = torch.nn.Linear(2, 4, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([
            [2.0, 0.0],
            [0.0, 2.0],
            [-2.0, 0.0],
            [0.0, -2.0]
        ]))
    return model


def main():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)

    model = _make_model()
    model.train()
    mapping = {0: [0, 1], 1: [2, 3]}
    eval_loaders = [
        DataLoader(
            TensorDataset(
                torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                torch.tensor([0, 1])
            ),
            batch_size=1,
            shuffle=True
        ),
        DataLoader(
            TensorDataset(
                torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),
                torch.tensor([2, 3])
            ),
            batch_size=1,
            shuffle=True
        )
    ]
    buffer = _make_buffer()

    buffer_before = copy.deepcopy(buffer.data)
    counts_before = read_buffer_counts(buffer)
    counts_after = read_buffer_counts(buffer)
    assert counts_before == counts_after
    _assert_buffer_equal(buffer.data, buffer_before)

    params_before = [parameter.detach().clone() for parameter in model.parameters()]
    rng_before = _capture_rng()
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        task0_metrics = run_thirdmod_diagnostic(
            model, eval_loaders, mapping, current_task=0, buffer=buffer, max_samples=2)
        task1_metrics = run_thirdmod_diagnostic(
            model, eval_loaders, mapping, current_task=1, buffer=buffer, max_samples=2)
    rng_after = _capture_rng()

    output = stdout.getvalue()
    assert '[THIRDMOD-DIAG]' in output
    assert task0_metrics['old_to_new_error_rate'] is None
    assert task0_metrics['new_to_old_error_rate'] is None
    assert task0_metrics['old_logits_mean'] is None
    assert task0_metrics['new_logits_mean'] is None
    assert task0_metrics['logit_gap_new_minus_old'] is None
    assert task1_metrics['old_to_new_error_rate'] is not None
    assert task1_metrics['new_to_old_error_rate'] is not None
    assert task1_metrics['logit_gap_new_minus_old'] is not None
    assert sum(task1_metrics['pred_task_hist'].values()) == 4
    assert model.training
    assert all(
        torch.equal(before, after)
        for before, after in zip(params_before, model.parameters())
    )
    _assert_buffer_equal(buffer.data, buffer_before)
    _assert_rng_equal(rng_before, rng_after)

    print(output, end='')
    print('synthetic_thirdmod_diagnostic_smoke: PASS')


if __name__ == '__main__':
    main()
