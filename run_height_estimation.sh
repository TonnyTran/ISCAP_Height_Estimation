#!/bin/bash
#####
# Author:   Tran The Anh
# Date:     Sept 2021
# Project:  ISCAP - Identifying Speaker Characteristics through Audio Profiling
# Topic:    Height Estimation
# Licensed: Nanyang Technological University
#####

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

program=$1          # Choose program that you want to run {1, 2}
running=$2          # {TRAINING, TESTING} -> select to monitor the program 
band=$3             # {wideband, narrowband} change this parameter to run the program on wideband or narrowband data
gender_input=$4     # {nogender, withgender} -> without or with gender as an input

if [ -z "${program}" ]; then
    echo 'Warning: Please input the program that you want to run which is in {1, 2}.'
    echo 'E.g. bash run_height_estimation.sh 1 => run program 1'
    exit 1
fi

#### Model 1: LSTM + Cross_Attention + MSE_Loss | FBank Features | MultiTask Estimation (both age & height)
if [ ${program} -eq 1 ]; then
    python code/height_age_multitask_run.py $running $band $gender_input     # we can change the nb of GPU using for training by modifying line 55 in code/height_age_multitask_run.py
fi

#### Model 2: LSTM + Cross_Attention + Triplet & MSE_Loss | FBank Features | Height Estimation
if [ ${program} -eq 2 ]; then
    python code/height_triplet_mse_run.py $running $band $gender_input      # we can change the nb of GPU using for training by modifying line 55 in code/height_triplet_mse_run.py
fi