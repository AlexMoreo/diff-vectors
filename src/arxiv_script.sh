#!/usr/bin/env bash
set -x

source activate torch

dataset="arxiv"

for seed in {0..9} ; do

    # Attribution
    authors="-1"
    common="--docs_by_author -1 --n_authors $authors"
    pickle="../pickles/$dataset-A$authors-D-1-S$seed.pickle"
    #for method in LR PairLRknn PairLRlinear LRbin ; do
    for method in  LRbin ; do
        python3 main.py $dataset $method LR $common --logfile=../log/results_$dataset-$method.csv --pickle $pickle --seed $seed
    done
    rm $pickle

    # SAV
#    common="--docs_by_author -1 --n_authors -1"
#    python same_author_verification.py $dataset LR --logfile ../log-sav/SAV_$dataset.csv $common --n_open_authors 50 --seed $seed

done

