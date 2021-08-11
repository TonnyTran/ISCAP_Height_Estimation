#!/bin/bash

# Copyright 2019 IIIT-Bangalore (Shreekantha Nadig)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh
. ./cmd.sh
# We can control the program flow by changing start and stop stage
stage=3                 # start stage
stop_stage=3            # stop stage
# Raw TIMIT dataset location. If TIMIT dataset is available on your computer, you can change raw_data to your dataset location
timit=`pwd`/data/lisa/data/timit/raw/TIMIT         # Raw TIMIT data directory (after extracted from .zip file)
spkInfor=$raw_data/DOC/SPKRINFO.TXT
trans_type=char

data=data               # data directory
dumpdir=$data/dump      # Dump data directory, which stores data used for training, validating and testing
do_delta=false

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

#### Download TIMIT dataset: TIMIT.zip >> save to folder data >> unzip TIMIT dataset
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "____________State 0: Download and extract raw TIMIT dataset____________"
    mkdir $data 
    wget https://ndownloader.figshare.com/files/10256148 -O $data/TIMIT.zip
    zipped_data=data/TIMIT.zip
    unzip $zipped_data
    echo "____________State 0: Successfully download and extract raw TIMIT dataset____________"
fi

#### Process TIMIT data using script from Kaldi.
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "____________State 1: Data preparation____________"
    # This step creates some files which are stored in data/local/data
    local/timit_data_prep_includeSA_fullTestSet.sh ${timit} ${trans_type} || exit 1
    # Prepare train and test data. This step creates data/train and data/test
    local/timit_format_data_includeSA_fullTestSet.sh
    echo "____________State 1: Successfully Data preparation____________"
fi

#### To train a neural network, we need training and validation sets.
#Therefore, we need to split data/train into 2 sets: data/trainNet_sp for training and data/valid for validation ####
# Utterances of 5% speakers in train set are used to create valid set (used for vadilating) - Vadilation set
# Utterances of 95% speakers in train set are used to create trainNet set
#=> do speed perturbation to form trainNet_sp (used for training) - Training set
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    echo "____________State 2: Split train set into Training set and Validation set____________"
    utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 $data/train $data/trainNet $data/valid
    utils/data/perturb_data_dir_speed_3way.sh $data/trainNet $data/trainNet_sp  ### speed perturbation
    echo "____________State 2: Successfully split train set into Training set and Validation set____________"
fi

train_set=trainNet_sp
train_dev=valid
recog_set=test

### Now extract features and dump features ######
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "____________State 3: Extract features and dump features____________"    
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    fbankdir=$data/fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in $train_set $train_dev ${recog_set}; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
        $data/${x} exp/make_fbank/${x} ${fbankdir}
    done

    # compute global CMVN
    compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark

    feat_train_dir=${dumpdir}/${train_set}; mkdir -p ${feat_train_dir}
    feat_dev_dir=${dumpdir}/${train_dev}; mkdir -p ${feat_dev_dir}
    # dump features
    dump.sh --cmd "$train_cmd" --nj 20 --do_delta false \
    data/${train_set}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/$train_set ${feat_train_dir}
    dump.sh --cmd "$train_cmd" --nj 20 --do_delta false \
    data/${train_dev}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/$train_dev ${feat_dev_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 20 --do_delta false \
        data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
        ${feat_recog_dir}
    done
    echo "____________State 3: Successfully extract features and dump features____________"
fi

# #### Now extract the dictionary and json files for training ####
# # dict=data/lang_1char/${train_set}_units.txt
# # echo "dictionary: ${dict}"
# if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
#     ### Task dependent. You have to check non-linguistic symbols used in the corpus.
#     echo "stage 2: Dictionary and Json Data Preparation"
#     mkdir -p data/lang_1char/
#     echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
#     text2token.py -s 1 -n 1 data/${train_set}/text --trans_type ${trans_type} | cut -f 2- -d" " | tr " " "\n" \
#     | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
#     wc -l ${dict}

#     # make json labels
#     data2json.sh --feat ${feat_train_dir}/feats.scp --trans_type ${trans_type} \
#     data/${train_set} ${dict} > ${feat_train_dir}/data.json
#     data2json.sh --feat ${feat_dev_dir}/feats.scp --trans_type ${trans_type} \
#     data/${train_dev} ${dict} > ${feat_dev_dir}/data.json
#     for rtask in ${recog_set}; do
#         feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
#         data2json.sh --feat ${feat_recog_dir}/feats.scp --trans_type ${trans_type} \
#         data/${rtask} ${dict} > ${feat_recog_dir}/data.json
#     done
# fi