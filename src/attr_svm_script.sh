#!/usr/bin/env bash
set -x

source activate torch

learner=SVM

for seed in {0..9} ; do

  for dataset in arxiv victorian pan2011 imdb62 ; do

    if [ $dataset == 'arxiv' ]; then
      authors=100
      docs=-1
    else
      authors=25
      docs=50
    fi

    common="--n_authors $authors --docs_by_author $docs --seed $seed"
    results="../log-aa-svm-r1/results_$dataset.csv"
    pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"

    for method in PairLRlinear LR ; do  # LRbin PairLRknn
      python3 authorship_attribution.py $dataset $method $learner $common --logfile $results --pickle $pickle
    done

  done

done



