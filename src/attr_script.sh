#!/usr/bin/env bash
set -x

source activate torch

#raw="--rawfreq"
raw=""

for seed in {0..9} ; do
    for dataset in victorian ; do #pan2011 imdb62  ; do
        for authors in 5 10 15 20 25 ; do
            for docs in 5 10 15 20 25 30 35 40 45 50 ; do
                pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"
#                for method in LR PairLRknn PairLRlinear LRbin ; do
                for method in LRbin ; do
                    common="--n_authors=$authors --docs_by_author=$docs --seed $seed --pickle $pickle $raw"
                    python main.py $dataset $method LR $common --logfile=../log/results_$dataset-$method.csv --seed $seed
                done
                rm $pickle
            done
        done
    done
done

#python main.py imdb62 PairLRlinear LR --n_authors 20 --docs_by_author=50 --seed 0 --pickle ../pickles/imdb62_20_50.pickle --logfile=../log/tmp.csv