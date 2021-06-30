#!/usr/bin/env bash
set -x

source activate torch

docs_by_author=50

#log="--logfile ../log-sav/SAV_dist.csv"

learner=SVM

for seed in {0..9} ; do
    for dataset in victorian pan2011 imdb62 ; do
        log="--logfile ../log-sav/SAV_$dataset.csv"
        for authors in 5 10 15 20 25 ; do
            python same_author_verification.py $dataset $learner $log --docs_by_author $docs_by_author --n_authors $authors --n_open_authors $authors --seed $seed
        done
    done
done


for seed in {0..9} ; do
    dataset=pan2011
    log="--logfile ../log-sav/SAV_pan2011.csv"
    docs_by_author=40
    authors=25
    python same_author_verification.py $dataset $learner $log --docs_by_author $docs_by_author --n_authors $authors --n_open_authors $authors --seed $seed

    dataset=victorian
    log="--logfile ../log-sav/SAV_victorian.csv"
    docs_by_author=50
    authors=25
    nopenauthors=20
    python same_author_verification.py $dataset $learner $log --docs_by_author $docs_by_author --n_authors $authors --n_open_authors $nopenauthors --seed $seed
done
