#!/usr/bin/env bash
set -x

source activate torch

AA="python authorship_attribution.py"

#raw="--rawfreq"
raw=""

for seed in {0..9} ; do
    for method in LR LRbin PairLRknn PairLRlinear ; do

      # experiments for arxiv
      pickle="../pickles/arxiv-A-1-D-1-S$seed$raw.pickle"
      common="--n_authors=-1 --docs_by_author=-1 --seed $seed --pickle $pickle $raw"
      $AA arxiv $method LR $common --logfile=../log-aa-r1/results_arxiv-$method.csv --seed $seed

      # experiments for the rest of the datasets
      for dataset in victorian pan2011 imdb62 ; do
          for authors in 5 10 15 20 25 ; do
#              for docs in 5 ; do # 10 15 20 25 30 35 40 45 50 ; do
              for docs in 10 20 30 40 50 ; do
                  pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"
                  common="--n_authors=$authors --docs_by_author=$docs --seed $seed --pickle $pickle $raw"
                  $AA $dataset $method LR $common --logfile=../log-aa-r1/results_$dataset-$method.csv --seed $seed
#                  rm $pickle
              done
          done
      done
    done
done

