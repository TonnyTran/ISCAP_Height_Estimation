#!/bin/bash
#####
# Author:   Tran The Anh
# Date:     Sept 2021
# Project:  ISCAP - Identifying Speaker Characteristics through Audio Profiling
# Topic:    Height Estimation
# Licensed: Nanyang Technological University
#####

. ./path.sh
. ./cmd.sh

# Note for Google Colab: Set USER environment variable. USER environment variable is not set in Google Colab. 
# If you are running this notebook locally, then you don't need to run these line
# import os
# os.environ['USER'] = 'your_username'
# !echo $USER

# We can control the program flow by changing start and stop stage
steps=0-3         # to run step 0 to step 3, use steps=0-3

# Raw TIMIT dataset location. If TIMIT dataset is available on your computer, you can change raw_data to your dataset location
timit=`pwd`/data/data/lisa/data/timit/raw/TIMIT         # Raw TIMIT data directory (after extracted from .zip file)
spkInfor=$timit/DOC/SPKRINFO.TXT
trans_type=char

data=data               # data directory 
dumpdir=$data/dump      # Dump data directory, which stores data used for training, validating and testing
do_delta=false

train_set=train
valid_set=valid
test_set=test

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}
        elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }}
      print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    declare step$index=1
  done
fi

#### Download TIMIT dataset: TIMIT.zip >> save to folder data >> unzip TIMIT dataset
if [ ! -z $step00 ]; then
    echo "____________Step 0: Download and extract raw TIMIT dataset____________"
    # wget https://ndownloader.figshare.com/files/10256148 -O raw_data/TIMIT.zip
    zipped_data=raw_data/TIMIT.zip
    mkdir $data
    unzip $zipped_data -d $data
    echo "____________Step 0: Successfully download and extract raw TIMIT dataset____________"
fi

#### Process TIMIT data using script from Kaldi.
if [ ! -z $step01 ]; then
    echo "____________Step 1: Data preparation____________"
    # This step creates some files which are stored in data/local/data
    local/timit_data_prep_includeSA_fullTestSet.sh ${timit} ${trans_type} || exit 1
    # Prepare train and test data. This step creates data/original/train and data/original/test
    local/timit_format_data_includeSA_fullTestSet.sh
    echo "____________Step 1: Successfully Data preparation____________"
fi

#### To train a neural network, we need training and validation sets.
#Therefore, we need to split data/train into 2 sets: data/trainNet_sp for training and data/valid for validation ####
# Utterances of 5% speakers in train set are used to create valid set (used for vadilating) - Vadilation set
# Utterances of 95% speakers in train set are used to create trainNet set
#=> do speed perturbation to form trainNet_sp (used for training) - Training set
if [ ! -z $step02 ]; then
    echo "____________Step 2: Split train set into Training set and Validation set____________"
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $data/original/train $data/original/trainNet $data/original/valid
    local/perturb_data_dir_speed_tempo.sh $data/original/trainNet $data/original/trainNet_sp_tempo  ### speed perturbation
    cp -avr $data/original/trainNet_sp_tempo $data/train
    cp -avr $data/original/valid $data/valid
    cp -avr $data/original/test $data/test
    echo "____________Step 2: Successfully split train set into Training set and Validation set____________"
fi

### Now extract features and dump features ######
if [ ! -z $step03 ]; then
    echo "____________Step 3: Extract features and dump features (Wideband)____________"    
    ### Task dependent. You have to design training and valid sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    fbankdir=$data/temp/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in $train_set $valid_set ${test_set}; do
        steps/make_fbank_pitch.sh --cmd "$cmd" --nj 20 --write_utt2num_frames true \
        $data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    feat_train_dir=${dumpdir}/${train_set}; mkdir -p ${feat_train_dir}
    feat_valid_dir=${dumpdir}/${valid_set}; mkdir -p ${feat_valid_dir}
    # dump features
    dump.sh --cmd "$cmd" --nj 20 --do_delta false \
    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/$train_set ${feat_train_dir}
    dump.sh --cmd "$cmd" --nj 20 --do_delta false \
    data/${valid_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/$valid_set ${feat_valid_dir}
    for rtask in ${test_set}; do
        feat_recog_dir=${dumpdir}/${rtask}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$cmd" --nj 20 --do_delta false \
        data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
        ${feat_recog_dir}
    done
    echo "____________Step 3: Successfully extract features and dump features (Wideband)____________"
fi

