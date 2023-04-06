#!/usr/bin/env bash
set -x

source activate torch

authors=25
docs=50

for seed in {0..9} ; do

  for dataset in victorian pan2011 imdb62 ; do

    common="--n_authors $authors --docs_by_author $docs --seed $seed"
    model="../models/roberta-$dataset-A$authors-D$docs-S$seed"
    results="../log-aa-bert-r1/results_$dataset.csv"
    pickle="../data/bert/$dataset-A$authors-D$docs-S$seed.pickle"

#    python3 bert-AA-finetuning.py $dataset --modelname $model $common

#    python3 bert-AA-classification.py $dataset RoBERTa $model $common --logfile $results

#    python3 bert-AA-embed-dataset.py $dataset $model $pickle $common


    for method in LR LRbin PairLRknn PairLRlinear ; do
      python3 authorship_attribution.py $dataset $method LR $common --logfile $results --pickle $pickle
    done

    exit
    
  done

done

