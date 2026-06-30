@echo off
setlocal

set "PROJECT_DIR=E:\eawqaar\csrel\ChunkChunk_mainline_clean_260617"
set "RUN_TAG=cifar100_cl_seed0_full_snapshot_q35_q55gate_rcgr_w8a4_time_260630"
set "LOG_DIR=E:\eawqaar\csrel\logs\%RUN_TAG%"
set "RESULT_DIR=E:\eawqaar\csrel\results\%RUN_TAG%"
set "DATA_PATH=E:\eawqaar\csrel\data"
set "PYTHON_EXE=E:\eawqaar\env\csrel_py3918\python.exe"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
if not exist "%RESULT_DIR%" mkdir "%RESULT_DIR%"

set "LOG_FILE=%LOG_DIR%\%RUN_TAG%.log"
set "TIME_FILE=%LOG_DIR%\%RUN_TAG%_time.txt"

cd /d "%PROJECT_DIR%"

set CUBLAS_WORKSPACE_CONFIG=:4096:8

powershell -NoProfile -ExecutionPolicy Bypass -Command "$start=Get-Date; $line=$start.ToString('yyyy-MM-dd HH:mm:ss.fff'); Set-Content -Path $env:TIME_FILE -Value $line; Add-Content -Path $env:LOG_FILE -Value ('RUN_START = ' + $line)"

"%PYTHON_EXE%" -u offline_continual_learning.py ^
  --local_path "%RESULT_DIR%" ^
  --dataset splitcifar100 ^
  --setting greedy ^
  --data_path "%DATA_PATH%" ^
  --buffer_size 200 ^
  --alpha 4.0 ^
  --beta 0.0 ^
  --lr 0.02 ^
  --epochs 100 ^
  --batch_size 32 ^
  --mem_batch_size 32 ^
  --use_cuda 1 ^
  --opt_type sgd ^
  --seed 0 ^
  --slt_wo_aug 0 ^
  --holdout_set sub ^
  --replay_mode sub ^
  --use_bn 1 ^
  --limit_per_task 5000 ^
  --runner_type coreset ^
  --update_mode coreset ^
  --extra_data "" ^
  --ref_train_epoch 10 ^
  --selection_steps 40 ^
  --cur_train_steps 7 ^
  --ref_train_lr 3e-3 ^
  --cur_train_lr 5e-3 ^
  --buffer_type coreset ^
  --ref_sample_per_task 0 ^
  --aug_type greedy ^
  --selection_strategy rel_gss_anchor_replace_hist_top4_prob ^
  --selection_chunk_size 4 ^
  --gss_anchor_replace_window 8 ^
  --gss_anchor_replace_anchor_size 4 ^
  --gss_anchor_replace_sim_threshold 0.90 ^
  --gss_anchor_replace_prob_conservativeness 1.0 ^
  --gss_grad_layer classifier ^
  --rel_ref_mode snapshot ^
  --snapshot_reduce quantile ^
  --snapshot_quantile 0.35 ^
        --snapshot_dynamic_calibration ^
        --snapshot_calib_beta 0.55 ^
        --snapshot_calib_use_confidence ^
        --snapshot_calib_use_variability ^
        --snapshot_calib_rho_max 0.20 ^
  --snapshot_agreement_lambda 0.0 ^
  >> "%LOG_FILE%" 2>&1

set "EXIT_CODE=%ERRORLEVEL%"

powershell -NoProfile -ExecutionPolicy Bypass -Command "$timeFile=$env:TIME_FILE; $logFile=$env:LOG_FILE; $exitCode=[int]$env:EXIT_CODE; $end=Get-Date; $start=[datetime](Get-Content $timeFile | Select-Object -First 1); $dur=$end-$start; $lines=@(('RUN_END = ' + $end.ToString('yyyy-MM-dd HH:mm:ss.fff')), ('RUN_DURATION = ' + $dur.ToString('hh\:mm\:ss')), ('RUN_DURATION_MINUTES = ' + [math]::Round($dur.TotalMinutes,2)), ('EXIT_CODE = ' + $exitCode)); $lines | Add-Content -Path $timeFile; $lines | Add-Content -Path $logFile"

exit /b %EXIT_CODE%




