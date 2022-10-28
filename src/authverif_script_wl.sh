#!/usr/bin/env bash
set -x

#wl stands for "with labels known", i.e., using all n labels for generating positives, but then converting the problem
#to binary

source activate torch

#raw="--rawfreq"
raw=""

for seed in {0..9} ; do
    for dataset in victorian pan2011 imdb62  ; do
        for authors in 10 ; do
            for docs in 10 50 ; do
                pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"
                for method in LR PairLRlinearbin_wl PairLRknnbin_wl; do # PairLRknnbin_wl
                    common="--n_authors=$authors --docs_by_author=$docs --seed $seed --pickle $pickle $raw"
                    python authorship_verification_withauthorsknown.py $dataset $method LR $common --logfile=../logbin/results_$dataset-$method-wl.csv
                done
                rm $pickle
            done
        done
    done
done

#python main.py imdb62 PairLRlinear LR --n_authors 20 --docs_by_author=50 --seed 0 --pickle ../pickles/imdb62_20_50.pickle --logfile=../log/tmp.csv