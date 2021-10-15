#!/bin/bash

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train and test data"
srcdir=data/local/data
outdir=data/original
# lmdir=data/local/nist_lm
# tmpdir=data/local/lm_tmp
# lexicon=data/local/dict/lexicon.txt
# mkdir -p $tmpdir

for x in train test; do
    mkdir -p $outdir/$x
    cp $srcdir/${x}_wav.scp $outdir/$x/wav.scp || exit 1;
    cp $srcdir/$x.text $outdir/$x/text || exit 1;
    cp $srcdir/$x.spk2utt $outdir/$x/spk2utt || exit 1;
    cp $srcdir/$x.utt2spk $outdir/$x/utt2spk || exit 1;
    utils/filter_scp.pl $outdir/$x/spk2utt $srcdir/$x.spk2gender > $outdir/$x/spk2gender || exit 1;
    cp $srcdir/${x}.stm $outdir/$x/stm
    cp $srcdir/${x}.glm $outdir/$x/glm
    utils/validate_data_dir.sh --no-feats $outdir/$x || exit 1
done
