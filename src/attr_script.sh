#!/usr/bin/env bash
set -x

source activate torch

AA="python authorship_attribution.py"

raw=""
learner="LR"
logdir="../log-aa"

for seed in {0..9} ; do
    for method in LR LRbin PairLRknn PairLRlinear ; do

      # experiments for arxiv
      pickle="../pickles/arxiv-A-1-D-1-S$seed$raw.pickle"
      common="--n_authors=-1 --docs_by_author=-1 --seed $seed --pickle $pickle $raw"
      $AA arxiv $method $learner $common --logfile=$logdir/results_arxiv-$method.csv --seed $seed

      # experiments for the rest of the datasets
      for dataset in victorian pan2011 imdb62 ; do
          for authors in 5 10 15 20 25 ; do
              for docs in 5 10 15 20 25 30 35 40 45 50 ; do
                  pickle="../pickles/$dataset-A$authors-D$docs-S$seed$raw.pickle"
                  common="--n_authors=$authors --docs_by_author=$docs --seed $seed --pickle $pickle $raw"
                  $AA $dataset $method $learner $common --logfile=$logdir/results_$dataset-$method.csv --seed $seed
              done
          done
      done
    done
done

# see also src/utils/grid_plot_attr.py for generating grid plots for victorian pan2011 imdb62