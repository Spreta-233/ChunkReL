@echo off
setlocal

cd /d E:\eawqaar\csrel\CSReL-Coreset-CL-rd

set RUN_TAG=csrel_ccrf_cifar10_cl_seed0_chunk4_sim085_20260527
set LOG_DIR=E:\eawqaar\csrel\logs\%RUN_TAG%
set RESULT_DIR=E:\eawqaar\csrel\results\%RUN_TAG%\run

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

call E:\anaconda3\Scripts\activate.bat E:\eawqaar\env\csrel_py3918
if errorlevel 1 exit /b %errorlevel%

python -u offline_continual_learning.py ^
  --local_path "%RESULT_DIR%" ^
  --dataset splitcifar ^
  --setting greedy ^
  --data_path E:\eawqaar\csrel\data ^
  --buffer_size 200 ^
  --epochs 400 ^
  --batch_size 256 ^
  --mem_batch_size 32 ^
  --alpha 20.0 ^
  --beta 0.0 ^
  --lr 1e-3 ^
  --use_cuda 1 ^
  --opt_type adam ^
  --runner_type coreset ^
  --update_mode coreset ^
  --buffer_type coreset ^
  --holdout_set sub ^
  --replay_mode full ^
  --use_bn 0 ^
  --limit_per_task 1000 ^
  --extra_data= ^
  --ref_train_epoch 150 ^
  --selection_steps 200 ^
  --cur_train_steps 100 ^
  --ref_train_lr 3e-3 ^
  --cur_train_lr 1e-2 ^
  --ref_sample_per_task 0 ^
  --aug_type greedy ^
  --memory_allocation equal ^
  --selection_strategy rel_ccrf ^
  --selection_chunk_size 4 ^
  --ccrf_sim_threshold 0.85 ^
  --seed 0 > "%LOG_DIR%\detail.log" 2>&1

exit /b %errorlevel%
