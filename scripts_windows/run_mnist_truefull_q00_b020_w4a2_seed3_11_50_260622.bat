@echo off
setlocal enabledelayedexpansion

set "PROJECT_DIR=E:\eawqaar\csrel\ChunkChunk_mainline_clean_260617"
set "PYTHON_EXE=E:\eawqaar\env\csrel_py3918\python.exe"
set "DATA_PATH=E:\eawqaar\csrel\data"
set "LOG_BASE=E:\eawqaar\csrel\logs"
set "RESULT_BASE=E:\eawqaar\csrel\results"

cd /d "%PROJECT_DIR%"

echo ============================================================
echo MNIST OUR TRUE FULL q00-b020 w4a2 paired seeds
echo Method: Snapshot q0 + dynamic calibration + LLR gate + protected lowvar tail + GSS/R-CGR
echo Seeds: 3 11 50
echo Project: %PROJECT_DIR%
echo ============================================================

for %%S in (3 11 50) do (
    set "RUN_TAG=mnist_our_truefull_q00_b020_w4a2_seed%%S_260622"
    set "LOG_DIR=%LOG_BASE%\!RUN_TAG!"
    set "RESULT_DIR=%RESULT_BASE%\!RUN_TAG!"

    mkdir "!LOG_DIR!" 2>nul
    mkdir "!RESULT_DIR!" 2>nul

    echo.
    echo ============================================================
    echo START OUR MODEL seed %%S
    echo RUN_TAG=!RUN_TAG!
    echo LOG=!LOG_DIR!\!RUN_TAG!.log
    echo ============================================================

    "%PYTHON_EXE%" offline_continual_learning.py ^
        --local_path "!RESULT_DIR!" ^
        --data_path "%DATA_PATH%" ^
        --dataset splitmnist ^
        --setting greedy ^
        --buffer_size 100 ^
        --alpha 50.0 ^
        --beta 0.0 ^
        --lr 5e-4 ^
        --epochs 400 ^
        --batch_size 256 ^
        --mem_batch_size 32 ^
        --use_cuda 1 ^
        --opt_type adam ^
        --slt_wo_aug 0 ^
        --holdout_set sub ^
        --replay_mode full ^
        --use_bn 0 ^
        --limit_per_task 1000 ^
        --runner_type coreset ^
        --update_mode coreset ^
        --buffer_type coreset ^
        --extra_data "" ^
        --slt_mse_factor -1 ^
        --cur_train_steps 30 ^
        --ref_train_epoch 20 ^
        --selection_steps 100 ^
        --ref_train_lr 3e-3 ^
        --cur_train_lr 2e-2 ^
        --aug_type greedy ^
        --ref_sample_per_task 0 ^
        --seed %%S ^
        --memory_allocation equal ^
        --selection_strategy rel_gss_anchor_replace_hist_top4_prob ^
        --selection_chunk_size 2 ^
        --gss_anchor_replace_window 4 ^
        --gss_anchor_replace_anchor_size 2 ^
        --gss_anchor_replace_sim_threshold 0.90 ^
        --gss_anchor_replace_prob_conservativeness 1.0 ^
        --gss_grad_layer classifier ^
        --rel_ref_mode snapshot ^
        --snapshot_reduce quantile ^
        --snapshot_quantile 0.00 ^
        --snapshot_agreement_lambda 0.0 ^
        --snapshot_dynamic_calibration ^
        --snapshot_calib_beta 0.20 ^
        --snapshot_calib_rho_max 0.10 ^
        --snapshot_calib_tau 0.5 ^
        --snapshot_calib_norm_scope task ^
        --snapshot_calib_use_confidence ^
        --snapshot_calib_conf_gate linear_log_residual_protected_lowvar_tail ^
        --snapshot_calib_conf_log_lambda 0.2 ^
        --snapshot_calib_lowvar_tail_q 0.05 ^
        --snapshot_calib_lowvar_eta 0.05 ^
        --snapshot_calib_highconf_protect_q 0.80 ^
        --snapshot_calib_conf_gamma 1.0 ^
        --snapshot_calib_conf_eps 1e-6 ^
        > "!LOG_DIR!\!RUN_TAG!.log" 2>&1

    echo.
    echo FINISH OUR MODEL seed %%S

    echo.
    echo Checking real errors...
    powershell -NoProfile -Command "Select-String -Path '!LOG_DIR!\!RUN_TAG!.log' -Pattern 'Traceback|RuntimeError|CUDA out of memory|AttributeError|ValueError|unrecognized arguments'"

    echo.
    echo Last final accuracy line:
    powershell -NoProfile -Command "Select-String -Path '!LOG_DIR!\!RUN_TAG!.log' -Pattern 'accuracies on testset after task 4 is' | Select-Object -Last 1"
)

echo.
echo ============================================================
echo ALL OUR TRUE FULL q00-b020 w4a2 seed3/11/50 FINISHED
echo ============================================================
pause