#!/bin/bash

docs_by_author=20
n_authors=10
learner=LR
mix_train=False


for dataset in victorian pan2011 imdb62 arxiv ; do
  log="--logfile ../log-sav/nomix_SAV_$dataset.csv"
  python same_author_verification.py $dataset $learner $log --mix_train $mix_train --docs_by_author $docs_by_author --n_authors $n_authors --n_open_authors $n_authors
done