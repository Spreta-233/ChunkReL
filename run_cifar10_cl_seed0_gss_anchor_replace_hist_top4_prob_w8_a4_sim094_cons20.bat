@echo off
setlocal

set PROJECT_DIR=E:\eawqaar\csrel\ChunkReL_gss_prob_conservative_run
set "RUN_TAG=csrel_gss_anchor_replace_hist_top4_prob_cifar10_cl_seed0_w8_a4_sim094_cons20_20260601"
set "LOG_DIR=E:\eawqaar\csrel\logs\%RUN_TAG%"
set "RESULT_DIR=E:\eawqaar\csrel\results\%RUN_TAG%"
set "DATA_PATH=E:\eawqaar\csrel\data"
set "PYTHON_EXE=E:\eawqaar\env\csrel_py3918\python.exe"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

cd /d "%PROJECT_DIR%"
set CUBLAS_WORKSPACE_CONFIG=:4096:8

"%PYTHON_EXE%" -u offline_continual_learning.py ^
  --local_path "%RESULT_DIR%" ^
  --dataset splitcifar ^
  --setting greedy ^
  --data_path "%DATA_PATH%" ^
  --buffer_size 200 ^
  --alpha 20.0 ^
  --beta 0.0 ^
  --lr 1e-3 ^
  --epochs 400 ^
  --batch_size 256 ^
  --mem_batch_size 32 ^
  --use_cuda 1 ^
  --opt_type adam ^
  --seed 0 ^
  --slt_wo_aug 0 ^
  --holdout_set sub ^
  --replay_mode full ^
  --use_bn 0 ^
  --limit_per_task 1000 ^
  --runner_type coreset ^
  --update_mode coreset ^
  --extra_data "" ^
  --ref_train_epoch 150 ^
  --selection_steps 200 ^
  --cur_train_steps 100 ^
  --ref_train_lr 3e-3 ^
  --cur_train_lr 1e-2 ^
  --buffer_type coreset ^
  --ref_sample_per_task 0 ^
  --aug_type greedy ^
  --selection_strategy rel_gss_anchor_replace_hist_top4_prob ^
  --selection_chunk_size 4 ^
  --gss_anchor_replace_window 8 ^
  --gss_anchor_replace_anchor_size 4 ^
  --gss_anchor_replace_sim_threshold 0.94 ^
  --gss_anchor_replace_prob_conservativeness 2.0 ^
  --gss_grad_layer classifier ^
  > "%LOG_DIR%\%RUN_TAG%.log" 2>&1

endlocal

