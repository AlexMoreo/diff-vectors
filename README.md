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
* [authorship_attribution.py](src/authorship_attribution.py) performs experiments on authorship attribution (AA), 
  i.e., in the task of deciding who, among a closed-set of candidate authors, is the author of a test document. 
  Experiments can be carried out using the methods Lazy AA, Stacked AA, STD-AA, and STD-Bin.
* [authorship_verification.py](src/authorship_verification.py) performs experiments on authorship verification (AV), 
  i.e., in the task of deciding whether a candidate author is the true author of a given document or not.
  This script tackles the AV task in a _native_ setting, i.e., in which all we know at training time is whether
  a training document has been written by our candidate author (label 1) or not (label 0). In the latter case,
  we assume not know who else has written the document.
  Experiments can be carried out using the methods Lazy AA, Stacked AA, and STD-Bin.

Available dataset names to indicate include:
* imdb62: film reviews, [download](https://umlt.infotech.monash.edu/?page_id=266)
* pan2011: emails, [download](https://pan.webis.de/clef11/pan11-web/authorship-attribution.html)
* victorian: segments of novels, [download](https://archive.ics.uci.edu/ml/datasets/Victorian+Era+Authorship+Attribution)
* arxiv: abstract of papers, [download](https://doi.org/10.5281/zenodo.7404702). The dataset has been generated
  using the [fetch_arxiv.py](src%2Fdata%2Ffetch_arxiv.py) script

## Requirements

* scikit-learn
* pandas
* tqdm
* spacy:
    * python -m spacy download en_core_web_sm
* nltk
    *	python -m nltk.downloader punkt
    *	python -m nltk.downloader stopwords
* feedparser (for generating the arxiv dataset)



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
* [sav_script.sh](src/sav_script.sh): runs 10 experiments in datasets pan2011, victorian, and imdb62, varying the 
    number of authors and number of documents per author. This script also runs 10 experiments using all authors (i.e., 50 authors in closed-set, plus
    50 authors in open-set) in the dataset arxiv, considering all documents for each author.

See also:
* [scores_plot_sav.py](src/scores_plot_sav.py): generates each of the plots in Figures 3 and 5.
* [savplot_script.sh](src/savplot_script.sh): a script that invokes the [scores_plot_sav.py](src/scores_plot_sav.py)
    with specific arguments in order to generate all the plots shown in Figures 3 and 5..

## Authorship Attribution

The script [authorship_attribution.py](src/authorship_attribution.py) can be executed using the following syntax:

```
usage: authorship_attribution.py [-h] [--n_authors N] [--docs_by_author N] [--k N] [--logfile LOGFILE] [--pickle PICKLE] [--seed SEED] [--force] [--rawfreq] dataset method learner

This method performs experiments of Authorship Attribution

positional arguments:
  dataset             Name of the dataset to run experiments on (valid ones are {'victorian', 'arxiv', 'pan2011', 'imdb62'})
  method              Classification method (valid ones are {'PairLRknn', 'LR', 'PairLRlinear', 'LRbin'})
  learner             Base learner (valid ones are {'SVM', 'LR', 'MNB'})

optional arguments:
  -h, --help          show this help message and exit
  --n_authors N       Number of authors to extract
  --docs_by_author N  Number of texts by author to extract
  --k N               k for knn prob mean [-1, indicates self tunning]
  --logfile LOGFILE   File csv with results
  --pickle PICKLE     path to a file where to save/load a pickle of the dataset once built
  --seed SEED         random seed
  --force             do not check if the experiment has already been performed
  --rawfreq           use raw frequencies instead of tfidf

```

The exact batch of experiments we have carried out are executed using the scripts:
* [attr_script.sh](src%2Fattr_script.sh) run the experiments for datasets victorian pan2011 imdb62, varying the numver 
    of authors and documents per author, and for arxiv, using all documents and authors. Each experiment is repeated 10 
    times.
* [attr_svm_script.sh](src%2Fattr_svm_script.sh) run experiments using SVM+lri in place of LR as the learner device.
    Each experiment is carried out 10 times, but the number of authors is kept fixed to 25 authors and the number of 
    documents per author to 50 (in arxiv we use all documents and authors).

## Authorship Verification

The script [authorship_verification.py](src/authorship_verification.py) can be executed using the following syntax:

```
usage: authorship_verification.py [-h] [--n_authors N] [--docs_by_author N] [--k N] [--logfile LOGFILE] [--pickle PICKLE] [--seed SEED] [--force] [--rawfreq] dataset method learner

This method performs experiments of Authorship Verification

positional arguments:
  dataset             Name of the dataset to run experiments on (valid ones are {'imdb62', 'victorian', 'pan2011', 'arxiv', 'pan2020'})
  method              Classification method (valid ones are {'PairLRlinear', 'PairLRknn', 'PairLRknnbin', 'LR', 'PairLRlinearbin', 'LRbin'})
  learner             Base learner (valid ones are {'LR', 'SVM'})

optional arguments:
  -h, --help          show this help message and exit
  --n_authors N       Number of authors to extract
  --docs_by_author N  Number of texts by author to extract
  --k N               k for knn prob mean [-1, indicates self tunning]
  --logfile LOGFILE   File csv with results
  --pickle PICKLE     path to a file where to save/load a pickle of the dataset once built
  --seed SEED         random seed
  --force             do not check if the experiment has already been performed
  --rawfreq           use raw frequencies instead of tfidf

```

The exact batch of experiments we have carried out are executed using the scripts:

* [av_script.sh](src/av_script.sh): run the experiments reported in Section 4.7 for all datasets 