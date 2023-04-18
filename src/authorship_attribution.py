from utils.common import prepare_learner
from utils.result_manager import AttributionResult, check_if_already_performed
from feature_extraction.author_vectorizer import FeatureExtractor
from model.pair_classification import PairAAClassifier, PairSAVClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GridSearchCV
#from sklearn.calibration import CalibratedClassifierCV
from utils.calibration import CalibratedClassifierCV
from sklearn.svm import SVC, LinearSVC
from utils.evaluation import *
import os
import argparse
from time import time
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
import importlib
import pickle
import pathlib


def main():

    # load results set if exists, otherwise creates it
    csv = AttributionResult(opt.logfile)

    # dataset preparation
    Xtr, ytr, Xte, yte, vectorizer_time = prepare_dataset()

    # classifier training
    t_init = time()
    cls, ytr, yte = prepare_classifier(ytr=ytr, yte=yte)
    cls.fit(Xtr, ytr)
    train_time = time() - t_init

    # classifier test
    t_init = time()
    yte_ = cls.predict(Xte)
    test_time = time() - t_init

    # evaluation
    acc_attr, f1_attr_macro, f1_attr_micro = evaluation_attribution(yte, yte_)
    acc_verif, f1_verif_macro, f1_verif_micro = evaluation_verification(yte, yte_)

    # output
    num_el = cls.n_pairs_ if isinstance(cls, PairAAClassifier) else Xtr.shape[0]
    csv.append(opt.dataset, opt.seed, opt.n_authors, opt.docs_by_author, opt.method, num_el,
               vectorizer_time, train_time, test_time,
               acc_attr, f1_attr_macro, f1_attr_micro, f1_verif_micro, f1_verif_macro)
    csv.close()


def prepare_dataset():
    if opt.pickle and os.path.exists(opt.pickle):
        print('pre-built dataset exists, loading...')
        #(Xtr,ytr,Xte,yte,vectorizer_time) = pickle.load(open(opt.pickle, 'rb'))
        (Xtr, ytr, Xte, yte, _) = pickle.load(open(opt.pickle, 'rb'))
        vectorizer_time=0
    else:
        print('building dataset...')
        dataset_loader = getattr(importlib.import_module(f'data.fetch_{opt.dataset.lower()}'), opt.dataset.title())
        dataset = dataset_loader(n_authors=opt.n_authors, docs_by_author=opt.docs_by_author, n_open_set_authors=0, random_state=opt.seed)

        Xtr, ytr = dataset.train.data, dataset.train.target
        Xte, yte = dataset.test.data, dataset.test.target

        # feature extraction
        te_init = time()
        print(f'using raw_freq = {opt.rawfreq}')
        vectorizer = FeatureExtractor('english', cleaning=False, use_raw_frequencies=opt.rawfreq)
        Xtr = vectorizer.fit_transform(Xtr, ytr)
        Xte = vectorizer.transform(Xte, None)
        vectorizer_time = time() - te_init

        if opt.pickle:
            print('pickling built dataset...')
            pickle.dump((Xtr,ytr,Xte,yte,vectorizer_time), open(opt.pickle, 'wb'), pickle.HIGHEST_PROTOCOL)

    print(f'Xtr.shape={Xtr.shape}')
    print(f'Xte.shape={Xte.shape}')
    print(f'Authors={len(np.unique(ytr))}')

    return Xtr, ytr, Xte, yte, vectorizer_time


def prepare_classifier(ytr=None, yte=None):

    base_learner, maxinst = prepare_learner(opt.learner)

    if (opt.method in ['LR', 'LRbin']):
        # attribution and verification, where verification is resolved via single-label attribution first
        cls = base_learner

    elif (opt.method in ['PairLRknn', 'PairLRlinear']):
        pos = -1  # -1 stands for all
        neg = -1  # -1 stands for the same number as pos
        max = 50000
        sav = PairSAVClassifier(base_learner, pos, neg, max)
        if opt.method.startswith('PairLRknn'):
            if opt.k != -1: opt.method += f'-k{opt.k}'
            cls = PairAAClassifier(sav, cls_policy='knn', k=opt.k)
        elif opt.method.startswith('PairLRlinear'):
            cls = PairAAClassifier(sav, cls_policy='linear', learner=base_learner)

    return cls, ytr, yte


def assert_opt_in(option, option_name, valid):
    assert option in valid, f'unknown {option_name}, valid ones are {valid}'


if __name__ == '__main__':

    available_datasets = {'imdb62', 'pan2011', 'victorian', 'arxiv'}
    available_methods = {'LR', 'LRbin', 'PairLRknn', 'PairLRlinear'}
    available_learners = {'LR', 'SVM', 'MNB'}

    # Training settings
    parser = argparse.ArgumentParser(description='This method performs experiments of Authorship Attribution')
    parser.add_argument('dataset', type=str,
                        help=f'Name of the dataset to run experiments on (valid ones are {available_datasets})')
    parser.add_argument('method', type=str, help=f'Classification method (valid ones are {available_methods})')
    parser.add_argument('learner', type=str, help=f'Base learner (valid ones are {available_learners})')
    parser.add_argument('--n_authors', type=int, default=-1, metavar='N', help='Number of authors to extract')
    parser.add_argument('--docs_by_author', type=int, default=-1, metavar='N', help='Number of texts by author to extract')
    parser.add_argument('--k', type=int, default=-1, metavar='N', help='k for knn prob mean [-1, indicates self tunning]')
    parser.add_argument('--logfile', type=str, default='../log/results.csv', help='File csv with results')
    parser.add_argument('--pickle', type=str, default=None, help='path to a file where to save/load a pickle of the dataset once built')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--force', default=False, action='store_true', help='do not check if the experiment has already been performed')
    parser.add_argument('--rawfreq', default=False, action='store_true', help='use raw frequencies instead of tfidf')

    opt = parser.parse_args()

    assert_opt_in(opt.dataset, 'dataset', available_datasets)
    assert_opt_in(opt.method, 'method', available_methods)
    assert_opt_in(opt.learner, 'learner', available_learners)

    assert opt.force or not \
        check_if_already_performed(opt.logfile, opt.dataset, opt.seed, opt.n_authors, opt.docs_by_author, opt.method), \
        'experiment already performed'

    parentdir = pathlib.Path(opt.logfile).parent
    if parentdir:
        os.makedirs(parentdir, exist_ok=True)

    main()

