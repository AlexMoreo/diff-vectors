#!/usr/bin/env bash
set -x

source activate torch

dataset="arxiv"
#raw="--rawfreq"
raw=""

for seed in {0..9} ; do
    pickle="../pickles/$dataset-A50-D-1-S$seed.pickle"
    common="--n_authors=50 --n_open_authors=50 --docs_by_author=-1 --seed $seed --pickle $pickle $raw --logfile ../log-sav-r1/SAV_arxiv.csv"
    python3 same_author_verification.py $dataset LR $common
#    for method in LRbin PairLRknnbin PairLRlinearbin; do
#        python3 authorship_verification.py $dataset $method LR $common --logfile=../logAVfor10authors/results_AV_$dataset-$method.csv
#    done
#    rm $pickle

done
