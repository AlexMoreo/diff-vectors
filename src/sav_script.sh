#!/usr/bin/env bash
set -x

source activate torch

docs_by_author=50

#logdir="log-sav"
logdir="log-sav-r1"

learner=LR

# datasets victorian, pan2011, and imdb62 (exploration in 5 10 15 25 25 authors)
for seed in {0..9} ; do
    for dataset in victorian pan2011 imdb62 ; do
        log="--logfile ../$logdir/SAV_$dataset.csv"
        for authors in 5 10 15 20 25 ; do
            python same_author_verification.py $dataset $learner $log --docs_by_author $docs_by_author --n_authors $authors --n_open_authors $authors --seed $seed
        done
    done
done

# some configurations for 25 authors cannot be undertaken; these are the two exceptions we handle differently
for seed in {0..9} ; do
    dataset=pan2011
    log="--logfile ../$logdir/SAV_pan2011.csv"
    docs_by_author=40
    authors=25
    python same_author_verification.py $dataset $learner $log --docs_by_author $docs_by_author --n_authors $authors --n_open_authors $authors --seed $seed

    dataset=victorian
    log="--logfile ../$logdir/SAV_victorian.csv"
    docs_by_author=50
    authors=25
    nopenauthors=20
    python same_author_verification.py $dataset $learner $log --docs_by_author $docs_by_author --n_authors $authors --n_open_authors $nopenauthors --seed $seed
done

# for dataset arxiv, without the exploration by authors
dataset="arxiv"
for seed in {0..9} ; do
    pickle="../pickles/$dataset-A50-D-1-S$seed.pickle"
    log="--logfile ../$logdir/SAV_$dataset.csv"
    python3 same_author_verification.py $dataset --n_authors=50 --n_open_authors=50 --docs_by_author=-1 --pickle $pickle $log --seed $seed
done

# see also:
#   src/utils/arxiv_latex_tables_sav.py for creating the table for arxiv
#   src/utils/grid_plot_sav.py for creating the sav plots for victorian pan2011 imdb62