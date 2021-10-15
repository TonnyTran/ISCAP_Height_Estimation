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

# We can control the program flow by changing start and stop stage
steps=1-3         # to run step 1 to step 3, use steps=1-3
nj=10
data=data                           # data directory
dumpdir=$data/dump_narrowband       # Dump data directory, which stores data used for training, validating and testing
###------------------------------------------------
# end option
echo 
echo "$0 $@"
echo
set -e

. path.sh
. parse_options.sh || exit 1

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

train_set=train
test_set=test
valid_set=valid
source_dir=`pwd`/data
sample_rate=8000

# Down sample to 8K
if [ ! -z $step01 ]; then
  echo -e "____________Step 1: Down-sample to 8k data start @ `date`____________"
  for x in ${train_set} ${test_set} ${valid_set};do
    utils/fix_data_dir.sh $source_dir/$x
    utils/data/get_utt2dur.sh $source_dir/$x
    cat $source_dir/$x/utt2dur | awk '{print $1 " " $1 " 0 " $2}' > $source_dir/$x/segments

    utils/data/copy_data_dir.sh $source_dir/$x $source_dir/${x}_8k
    utils/data/resample_data_dir.sh $sample_rate $source_dir/${x}_8k
  done
  echo -e "____________Step 1: Successfully Down-sample to 8k data ended @ `date`____________"
fi

# codec
if [ ! -z $step02 ]; then
    echo -e "____________Step 2: Add codec start @ `date`____________"
    for x in ${train_set} ${test_set} ${valid_set};do
        utils/data/get_utt2dur.sh $source_dir/$x
        data_8k=$source_dir/${x}_8k
        data_codec=$source_dir/${x}_codec
        codec_list=./local/codec-list_full.txt
        ./local/copy-data-by-adding-codec.sh  --cmd "run.pl"  --steps 1,2,3 $data_8k $codec_list $data_codec || exit 1;
        cp $data_codec/tmp/{utt2dur,spk2utt,utt2spk} $data_codec
        rm -rf $data_8k
        utils/fix_data_dir.sh $data_codec
   done
    echo -e "____________Step 2: Successfully Add codec ended @ `date`____________"
fi

train_set=train_codec
valid_set=valid_codec
test_set=test_codec

### Now extract features and dump features ######
if [ ! -z $step03 ]; then
    echo "____________State 3: Extract features and dump features (Narrowband)____________"    
    ### Task dependent. You have to design training and valid sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    fbankdir=$data/temp/fbank_codec
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
    for x in $train_set $valid_set ${test_set}; do
        steps/make_fbank_pitch.sh --fbank_config conf/fbank_codec.conf --pitch_config conf/pitch_codec.conf --cmd "$cmd" --nj 20 --write_utt2num_frames true \
        $data/${x} exp/make_fbank_codec/${x} ${fbankdir}
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
    echo "____________State 3: Successfully extract features and dump features (Narrowband)____________"
fi
