# Diff-Vectors for Authorship Analysis

This repo contains the code for reproducing the experiments of the paper 
[Same or Different? Diff-Vectors for Authorship Analysis](https://arxiv.org/abs/2301.09862) (currently under submission).

The main files contained in the repo include:
* [same_author_verification.py](src/same_author_verification.py) performs intrinsic evaluation of different methods
  (DV-Bin, DV-2xAA, STD-CosDist, STD-CosDist, Impostors) in the task of same author verification (SAV), i.e., in the 
  task of deciding whether two documents have been written by the same author (albeit unknown) or not. The script
  carries out experiments in two settings:
  * closed-set SAV: test pairs are written by authors seen during training 
  * open-set SAV: test pairs are written by authors not seen during training
* 

Available dataset names to indicate include:
* imdb62: film reviews
* pan2011: emails
* victorian: segments of novels
* arxiv: abstract of papers 

## Same Author Verification
The script [same_author_verification.py](src/same_author_verification.py) can be executed using the following syntax:

```
usage: same_author_verification.py [-h] [--n_authors N] [--n_open_authors N] [--docs_by_author N] [--pickle PICKLE] [--logfile LOGFILE] [--seed SEED] [--rawfreq] DATASET LEARNER

This method performs experiments  on the same author verification (SAV) task considering the open-set and closed-set settings.

positional arguments:
  DATASET             Name of the dataset to run experiments on
  LEARNER             Base learner (valid ones are {'SVM', 'SGD', 'LR'})

optional arguments:
  -h, --help          show this help message and exit
  --n_authors N       Number of authors to extract
  --n_open_authors N  Number of authors for open set SAV
  --docs_by_author N  Number of texts by author to extract
  --pickle PICKLE     path to a file where to save/load a pickle of the dataset once built
  --logfile LOGFILE   File csv with results
  --seed SEED         random seed
  --rawfreq           use raw frequencies instead of tfidf
```

The exact batch of experiments we have carried out are executed using the scripts:
* [arxiv_script.sh](src%2Farxiv_script.sh) runs 10 experiments using all authors (i.e., 50 authors in closed-set, plus
    50 authors in open-set) in the dataset arxiv.
* [sav_script.sh](src/sav_script.sh): runs 10 experiments in datasets pan2011, victorian, and imdb62, varying the 
    number of authors and number of documents per author.




