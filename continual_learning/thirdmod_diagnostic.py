# -*-coding:utf8-*-

import json
import random

import numpy as np
import torch


LOG_PREFIX = '[THIRDMOD-DIAG]'


def _capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state().clone(),
        'cuda': None
    }
    if torch.cuda.is_available():
        state['cuda'] = [cuda_state.clone() for cuda_state in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def _capture_module_modes(model):
    return [(module, module.training) for module in model.modules()]


def _restore_module_modes(module_modes):
    for module, training in module_modes:
        module.training = training


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device('cpu')


def _normalise_mapping(task_class_mapping):
    mapping = {}
    for task_id, classes in task_class_mapping.items():
        mapping[int(task_id)] = [int(class_id) for class_id in classes]
    return mapping


def _iter_deterministic_batches(loader, max_samples):
    dataset = loader.dataset
    sample_count = len(dataset)
    if max_samples is not None and max_samples > 0:
        sample_count = min(sample_count, int(max_samples))
    batch_size = getattr(loader, 'batch_size', None) or 128
    batch_size = max(1, int(batch_size))
    for start in range(0, sample_count, batch_size):
        samples = [dataset[index] for index in range(start, min(start + batch_size, sample_count))]
        data = torch.stack([
            sample[0] if isinstance(sample[0], torch.Tensor) else torch.as_tensor(sample[0])
            for sample in samples
        ], dim=0)
        targets = torch.tensor([
            int(sample[1].item()) if isinstance(sample[1], torch.Tensor) else int(sample[1])
            for sample in samples
        ], dtype=torch.long)
        yield data, targets


def read_buffer_counts(buffer):
    """Read class/task counts without calling a sampling or training API."""
    if hasattr(buffer, 'get_diagnostic_counts'):
        counts = buffer.get_diagnostic_counts()
        return {
            'class_counts': dict(counts['class_counts']),
            'task_counts': dict(counts['task_counts'])
        }

    class_counts = {}
    task_counts = {}
    for task_id, task_data in enumerate(buffer.data):
        task_counts[int(task_id)] = len(task_data)
        for di in task_data:
            if len(di) not in [3, 4]:
                raise ValueError('Invalid buffer entry length')
            label = di[2]
            if isinstance(label, torch.Tensor):
                label = label.item()
            label = int(label)
            class_counts[label] = class_counts.get(label, 0) + 1
    return {
        'class_counts': class_counts,
        'task_counts': task_counts
    }


def _compute_metrics(model, eval_loaders, mapping, current_task, max_samples):
    seen_task_ids = [task_id for task_id in sorted(mapping) if task_id <= current_task]
    old_task_ids = [task_id for task_id in seen_task_ids if task_id < current_task]
    seen_classes = [class_id for task_id in seen_task_ids for class_id in mapping[task_id]]
    old_classes = [class_id for task_id in old_task_ids for class_id in mapping[task_id]]
    new_classes = list(mapping[current_task])
    class_to_task = {
        class_id: task_id
        for task_id, classes in mapping.items()
        for class_id in classes
    }
    if len(seen_classes) != len(set(seen_classes)):
        raise ValueError('task_class_mapping contains duplicate class ids')

    device = _model_device(model)
    seen_index = torch.tensor(seen_classes, dtype=torch.long, device=device)
    old_index = torch.tensor(old_classes, dtype=torch.long, device=device)
    new_index = torch.tensor(new_classes, dtype=torch.long, device=device)
    pred_task_hist = {task_id: 0 for task_id in sorted(mapping)}

    old_total = 0
    old_predicted_new = 0
    new_total = 0
    new_predicted_old = 0
    old_logit_sum = 0.0
    old_logit_count = 0
    new_logit_sum = 0.0
    new_logit_count = 0

    for task_id in seen_task_ids:
        for data, _targets in _iter_deterministic_batches(eval_loaders[task_id], max_samples):
            data = data.to(device)
            logits = model(data)
            if logits.ndim != 2:
                raise ValueError('Expected model logits with shape [batch, classes]')
            if max(seen_classes) >= logits.shape[1]:
                raise ValueError('task_class_mapping references a class outside model logits')
            predicted_seen_pos = logits.index_select(1, seen_index).argmax(dim=1)
            predicted_classes = seen_index.index_select(0, predicted_seen_pos)

            for predicted_class in predicted_classes.detach().cpu().tolist():
                predicted_task = class_to_task.get(int(predicted_class))
                if predicted_task is not None:
                    pred_task_hist[predicted_task] += 1

            if current_task == 0:
                continue
            if task_id < current_task:
                old_total += predicted_classes.numel()
                old_predicted_new += torch.isin(predicted_classes, new_index).sum().item()
                old_logits = logits.index_select(1, old_index)
                new_logits = logits.index_select(1, new_index)
                old_logit_sum += old_logits.sum().item()
                old_logit_count += old_logits.numel()
                new_logit_sum += new_logits.sum().item()
                new_logit_count += new_logits.numel()
            elif task_id == current_task:
                new_total += predicted_classes.numel()
                new_predicted_old += torch.isin(predicted_classes, old_index).sum().item()

    if current_task == 0:
        old_to_new_error_rate = None
        new_to_old_error_rate = None
        old_logits_mean = None
        new_logits_mean = None
        logit_gap = None
    else:
        old_to_new_error_rate = old_predicted_new / old_total if old_total else None
        new_to_old_error_rate = new_predicted_old / new_total if new_total else None
        old_logits_mean = old_logit_sum / old_logit_count if old_logit_count else None
        new_logits_mean = new_logit_sum / new_logit_count if new_logit_count else None
        logit_gap = (
            new_logits_mean - old_logits_mean
            if old_logits_mean is not None and new_logits_mean is not None else None
        )

    return {
        'task': current_task,
        'old_to_new_error_rate': old_to_new_error_rate,
        'new_to_old_error_rate': new_to_old_error_rate,
        'old_logits_mean': old_logits_mean,
        'new_logits_mean': new_logits_mean,
        'logit_gap_new_minus_old': logit_gap,
        'pred_task_hist': pred_task_hist
    }


def _format_value(value):
    if value is None:
        return 'NA'
    return '{:.8f}'.format(value)


def _print_metrics(metrics, mapping, buffer_counts, current_task):
    printable_mapping = {str(task_id): classes for task_id, classes in sorted(mapping.items())}
    print(LOG_PREFIX + ' task_class_mapping=' + json.dumps(printable_mapping, sort_keys=True))
    print(LOG_PREFIX + ' task=' + str(current_task))
    for key in [
            'old_to_new_error_rate',
            'new_to_old_error_rate',
            'old_logits_mean',
            'new_logits_mean',
            'logit_gap_new_minus_old']:
        print(LOG_PREFIX + ' ' + key + '=' + _format_value(metrics[key]))
    printable_hist = {str(task_id): count for task_id, count in sorted(metrics['pred_task_hist'].items())}
    print(LOG_PREFIX + ' pred_task_hist=' + json.dumps(printable_hist, sort_keys=True))

    seen_classes = [
        class_id
        for task_id in sorted(mapping)
        if task_id <= current_task
        for class_id in mapping[task_id]
    ]
    class_values = [buffer_counts['class_counts'].get(class_id, 0) for class_id in seen_classes]
    if class_values:
        class_min = min(class_values)
        class_max = max(class_values)
        class_mean = float(sum(class_values)) / len(class_values)
    else:
        class_min = 0
        class_max = 0
        class_mean = 0.0
    print(
        LOG_PREFIX + ' buffer_class_count_min/max/mean='
        + '{}/{}/{:.8f}'.format(class_min, class_max, class_mean)
    )
    printable_task_counts = {
        str(task_id): buffer_counts['task_counts'].get(task_id, 0)
        for task_id in sorted(mapping)
        if task_id <= current_task
    }
    print(LOG_PREFIX + ' buffer_task_count=' + json.dumps(printable_task_counts, sort_keys=True))


def run_thirdmod_diagnostic(
        model, eval_loaders, task_class_mapping, current_task, buffer, max_samples=1000):
    """Run read-only class-incremental diagnostics and restore all observable state."""
    mapping = _normalise_mapping(task_class_mapping)
    if current_task not in mapping:
        raise ValueError('current_task is missing from task_class_mapping')
    rng_state = _capture_rng_state()
    module_modes = _capture_module_modes(model)
    try:
        model.eval()
        with torch.no_grad():
            metrics = _compute_metrics(
                model=model,
                eval_loaders=eval_loaders,
                mapping=mapping,
                current_task=int(current_task),
                max_samples=max_samples
            )
            buffer_counts = read_buffer_counts(buffer)
        metrics['buffer_counts'] = buffer_counts
        _print_metrics(metrics, mapping, buffer_counts, int(current_task))
        return metrics
    finally:
        _restore_module_modes(module_modes)
        _restore_rng_state(rng_state)
