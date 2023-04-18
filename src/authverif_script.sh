#!/usr/bin/env bash
set -x

source activate torch

raw=""

for seed in {0..9} ; do
    for dataset in imdb62  victorian pan2011
        for authors in 10 ; do
            for docs in 50 ; do
                pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"
                for method in LRbin PairLRknnbin PairLRlinearbin; do
                    common="--n_authors=$authors --docs_by_author=$docs --seed $seed --pickle $pickle $raw"
                    python authorship_verification.py $dataset $method LR $common --logfile=../logAVfor10authors/results_AV_$dataset-$method.csv
                done
                rm $pickle
            done
        done
    done
done

python main.py imdb62 PairLRlinear LR --n_authors 20 --docs_by_author=50 --seed 0 --pickle ../pickles/imdb62_20_50.pickle --logfile=../log/tmp.csv
