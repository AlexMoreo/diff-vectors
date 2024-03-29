#!/usr/bin/env bash
set -x

source activate torch


docs_by_author=50

learner=LR

for seed in 1 ; do
    for dataset in victorian pan2011 imdb62 ; do
        #pkl="--pickle ../pickles/sav_plot_$dataset-$seed.pickle"
        pkl=../pickles/sav_plot_$dataset.pickle
        python3 scores_plot_sav.py $dataset $learner --pickle $pkl --docs_by_author 50 --n_authors 20 --n_open_authors 20 --seed $seed
    done

    dataset="arxiv"
    #pkl="--pickle ../pickles/sav_plot_$dataset-$seed.pickle"
    pkl=../pickles/sav_plot_$dataset.pickle
    python3 scores_plot_sav.py $dataset $learner --pickle $pkl --docs_by_author -1 --n_authors -1 --n_open_authors 50 --seed $seed
done