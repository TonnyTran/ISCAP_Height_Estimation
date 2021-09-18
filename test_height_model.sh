#####
# Author:   Tran The Anh
# Date:     Sept 2021
# Project:  ISCAP - Identifying Speaker Characteristics through Audio Profiling
# Topic:    Height Estimation
# Licensed: Nanyang Technological University
#####

#!/bin/bash
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

program=4                 # Choose program that you want to run {1, 2, 3 , 4}

#### Model 1: LSTM + Cross_Attention + MSE_Loss | FBank Features | Height Estimation
if [ ${program} -eq 1 ]; then
    python code/height_crossattn_test.py
fi

#### Model 2: LSTM + Cross_Attention + Center & MSE_Loss | FBank Features | Height Estimation
if [ ${program} -eq 2 ]; then
    python code/height_center_mse_test.py
fi

#### Model 3: LSTM + Cross_Attention + Triplet & MSE_Loss | FBank Features | Height Estimation
if [ ${program} -eq 3 ]; then
    python code/height_triplet_mse_test.py
fi

#### Model 4: LSTM + Cross_Attention + MAE_Loss | FBank Features | MultiTask Estimation (both age & height)
if [ ${program} -eq 4 ]; then
    python code/height_age_multitask_test.py
fi