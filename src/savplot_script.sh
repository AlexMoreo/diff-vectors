#!/usr/bin/env bash
set -x

source activate torch

#dataset="arxiv"

docs_by_author=50

learner=LR

for seed in {1..9} ; do
    for dataset in victorian pan2011 imdb62 ; do
        #pkl="--pickle ../pickles/sav_plot_$dataset-$seed.pickle"
        pkl="--pickle ../pickles/sav_plot_$dataset.pickle"
        python3 scores_plot_sav.py $dataset $learner $pkl --docs_by_author 50 --n_authors 20 --n_open_authors 20 --seed $seed
    done

    dataset="arxiv"
    #pkl="--pickle ../pickles/sav_plot_$dataset-$seed.pickle"
    pkl="--pickle ../pickles/sav_plot_$dataset.pickle"
    python3 scores_plot_sav.py $dataset $learner $pkl --docs_by_author -1 --n_authors -1 --n_open_authors 50 --seed $seed
done