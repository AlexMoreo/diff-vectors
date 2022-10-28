#!/usr/bin/env bash
set -x

source activate torch

#raw="--rawfreq"
raw=""

for seed in {0..9} ; do
    for dataset in victorian pan2011 imdb62  ; do
        for authors in 5 10 15 20 25 ; do
            for docs in 10 20 30 40 50 ; do
                pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"
                for method in LRbin PairLRknnbin PairLRlinearbin; do
                    common="--n_authors=$authors --docs_by_author=$docs --seed $seed --pickle $pickle $raw"
                    python authorship_verification.py $dataset $method LR $common --logfile=../logAV/results_AV_$dataset-$method.csv
                done
                rm $pickle
            done
        done
    done
done

#python main.py imdb62 PairLRlinear LR --n_authors 20 --docs_by_author=50 --seed 0 --pickle ../pickles/imdb62_20_50.pickle --logfile=../log/tmp.csv