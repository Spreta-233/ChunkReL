# Third-Module Diagnostic-Only Manifest

## Baseline

- Clean branch: `snapshot-llr-protlvt-rcgr`
- Clean commit: `ad112b5`
- Implementation branch: `thirdmod_diag_v1`
- Control mean: `42.03`

## Scope

This change adds diagnostic logging only. It does not implement BiC-lite,
EEIL-lite, SPG/RMAF-lite, or GPM-lite.

## Files

- `offline_continual_learning.py`: adds three opt-in CLI arguments and invokes
  diagnostics after buffer update and before `runner.next_task()`.
- `continual_learning/coreset_buffer.py`: adds a read-only count helper.
- `continual_learning/thirdmod_diagnostic.py`: computes and prints diagnostics.
- `synthetic_thirdmod_diagnostic_smoke.py`: verifies state preservation.
- `run_cifar10_cl_seed0_snapshot_q10_llr_protlvtail_rcgr_thirdmod_diag.bat`:
  runs the existing seed-0 control with diagnostic flags only.

## Diagnostic Semantics

- Prediction is class-incremental over all classes seen through the current task.
- Predicted classes are mapped to task IDs only after prediction.
- Evaluation samples are the deterministic first `thirdmod_diag_max_samples`
  samples from each seen task's test dataset.
- Buffer counts are read directly without replay sampling.
- Task 0 old-task metrics are reported as `NA`.
- All logs use the `[THIRDMOD-DIAG]` prefix.

## State Guarantees

- Runs under `model.eval()` and `torch.no_grad()`.
- Restores every module's prior train/eval flag.
- Saves and restores Python, NumPy, Torch CPU, and Torch CUDA RNG states.
- Does not update model parameters or buffer contents.
- With `--thirdmod_diag` disabled, no diagnostic module is imported or called.

## Formal Run

- BAT: `run_cifar10_cl_seed0_snapshot_q10_llr_protlvtail_rcgr_thirdmod_diag.bat`
- RUN_TAG: `csrel_q10_llr_protlvt_rcgr_thirdmod_diag_s0_260611_v2`
- Added flags:
  - `--thirdmod_diag`
  - `--thirdmod_diag_every_task`
  - `--thirdmod_diag_max_samples 1000`
