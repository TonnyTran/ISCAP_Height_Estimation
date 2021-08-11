#!/bin/bash

# Copyright 2019 IIIT-Bangalore (Shreekantha Nadig)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# cuda_cmd_all="/home/tungtest/slurm.pl --gpu 1 --nodelist=node07"
# cmd="run.pl"
# We can control the program flow by changing start and stop stage
program=$1

if [ ${program} =='' ]; then
    echo 'Warning: Please input the program that you want to run which is in {1, 2, 3, 4}.'
    echo 'E.g. bash test_height_estimation.sh 1 => run program 1'
    exit 1
fi

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