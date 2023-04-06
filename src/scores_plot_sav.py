import random

from authorship_verification import prepare_dataset, assert_opt_in
from same_author_verification import prepare_learner, load_dataset
from model.pair_impostors import PairImpostors, minmax, optimize_sigma, cosine
from same_author_verification import random_sample
from utils.common import get_verification_coordinates, get_parallel_slices
from utils.result_manager import AttributionResult, SAVResult
from feature_extraction.author_vectorizer import FeatureExtractor
from model.pair_classification import PairSAVClassifier, DistanceSAVClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from utils.evaluation import *
import os
import argparse
from tqdm import tqdm
import pickle
from sklearn.base import clone
import scipy
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from time import time


def main():

    #timingspath = '../timings'
    #os.makedirs(timingspath) if os.path.exists(timingspath) else None
    if os.path.exists('../timings/clock.csv'):
        timings = open('../timings/clock.csv', 'at')
    else:
        timings = open('../timings/clock.csv', 'wt')
        timings.write('Dataset\tmethod\ttime_tr\ttime_te\n')

    # dataset preparation
    n_authors = opt.n_authors
    n_open_authors = opt.n_open_authors
    docs_by_author = opt.docs_by_author
    docs_by_author_te = opt.docs_by_author
    Xtr, ytr, Xte, yte, Xte_out, yte_out = load_dataset(
        opt.dataset,
        n_authors=n_authors, docs_by_author=docs_by_author, n_open_set_authors=n_open_authors,
        seed=opt.seed, picklepath=opt.pickle, rawfreq=opt.rawfreq
    )

    # reset the random seed (of the dataset is generated then it uses random functions, if it is loaded, then not)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # classifier instantiation
    base_learner = prepare_learner(learner=opt.learner)

    max = 50000

    Xte, yte = random_sample(Xte, yte, cat_size=docs_by_author_te)
    Xte_out, yte_out = random_sample(Xte_out, yte_out, cat_size=docs_by_author_te)

    close_coordinates = get_verification_coordinates(yte, balanced=True, subsample=1000)
    open_coordinates = get_verification_coordinates(yte_out, balanced=True, subsample=1000)

    print(f'n_authors={n_authors} open_set_authors={n_open_authors}: docs_by_author={docs_by_author}')
    print(f'Xtr.shape={Xtr.shape} with clases={len(np.unique(ytr))}')
    print(f'Xte.shape={Xte.shape} with clases={len(np.unique(yte))}')
    print(f'Xte_out.shape={Xte_out.shape} with clases={len(np.unique(yte_out))}')

    plot_distance_method(DistanceSAVClassifier(-1, -1, max=max, metric='cosine'),
                         Xtr, ytr,
                         Xte, yte, close_coordinates,
                         Xte_out, yte_out, open_coordinates,
                         f"../pickles",
                         method='Dist-Cos', csv=timings)
    # plot_distance_method(DistanceSAVClassifier(-1, -1, max=max, metric='manhattan'),
    #                      Xtr, ytr,
    #                      Xte, yte, close_coordinates,
    #                      Xte_out, yte_out, open_coordinates,
    #                      f"../pickles",
    #                      method='Dist-L1', csv=timings)
    # plot_distance_method(DistanceSAVClassifier(-1, -1, max=max, metric='euclidean'),
    #                      Xtr, ytr,
    #                      Xte, yte, close_coordinates,
    #                      Xte_out, yte_out, open_coordinates,
    #                      f"../pickles",
    #                      method='Dist-L2', csv=timings)


    plot_impostors_method(Xtr, ytr, Xte_out, yte_out, open_coordinates, picklepath=f"../pickles", csv=timings)

    plot_pair_method(PairSAVClassifier(clone(base_learner), -1, -1, max=max),
                     Xtr, ytr, Xte, yte, close_coordinates, Xte_out, yte_out, open_coordinates,
                     picklepath=f"../pickles", csv=timings)


def plot_distance_method(dist_cls, Xtr, ytr,
                         Xte, yte, coordinates,
                         Xte_out, yte_out, open_coordinates,
                         picklepath, method, csv=None):
    picklefile = f"{picklepath}/dist_sav_{method}_{opt.dataset}_plot_seed{opt.seed}.pickle"
    if os.path.exists(picklefile):
        ((c_pos, c_neg), (o_pos,o_neg)) = pickle.load(open(picklefile, 'rb'))
    else:
        tinit=time()
        dist_cls.fit(Xtr, ytr)
        time_tr = time()-tinit

        c_pos, c_neg, _ = verification(dist_cls, Xte, yte, coordinates)
        o_pos, o_neg, time_te = verification(dist_cls, Xte_out, yte_out, open_coordinates)
        pickle.dump(((c_pos, c_neg), (o_pos,o_neg)), open(picklefile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        csv.write(f'{opt.dataset}\t{method}\t{time_tr}\t{time_te}\n')

    plot(c_pos, c_neg, method, path=f'../plot-sav/{opt.dataset}/{method}_close.png', decision=0.5)
    plot(o_pos, o_neg, method, path=f'../plot-sav/{opt.dataset}/{method}_open.png', decision=0.5)





def plot_impostors_method(Xtr, ytr, Xte_out, yte_out, open_coordinates, picklepath, csv):
    method = 'Impostors-open'
    picklefile = f"{picklepath}/impostors_sav_{method}_{opt.dataset}_plot_seed{opt.seed}.pickle"
    if os.path.exists(picklefile):
        positive, negative = pickle.load(open(picklefile, 'rb'))
    else:
        n, k = 10, 100  # parameters set according to the original Koppel's article
        m = 50  # m = 250 in Koppel's article, but they use many more potential candidates
        nDtr = Xtr.shape[0]
        if m > nDtr // 2:
            m = nDtr // 2
            n = min(10, m // 2)
        similarity = cosine
        tinit=time()
        sigma = optimize_sigma(Xtr, ytr, top_impostors=m, n_impostors=n, k=k, similarity=similarity)
        impostors_cls = PairImpostors(sigma=sigma, top_impostors=m, n_impostors=n, k=k, similarity=similarity)
        impostors_cls.fit(Xtr, ytr)
        time_tr = time() - tinit

        positive, negative, time_te = verification_via_impostors(impostors_cls, Xte_out, yte_out, open_coordinates)
        pickle.dump((positive, negative), open(picklefile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        csv.write(f'{opt.dataset}\t{method}\t{time_tr}\t{time_te}\n')

        # gamma victorian, pan2011, imdb62 0.005
        # arxiv 0.01
    decision=0.01 if opt.dataset == 'arxiv' else 0.005
    plot(positive, negative, method, path=f'../plot-sav/{opt.dataset}/{method}.png', logscale=False, decision=decision)
    plot(positive, negative, method, path=f'../plot-sav/{opt.dataset}/{method}-log.png', logscale=True, decision=decision)


def plot_pair_method(pair_cls, Xtr, ytr, Xte, yte, close_coordinates, Xte_out, yte_out, open_coordinates, picklepath, csv):
    method = 'DiffVectors'
    learner_st = '' if opt.learner == 'LR' else '_SVM'
    picklefile = f"{picklepath}/pair_sav_Pair-close{learner_st}_{opt.dataset}_seed{opt.seed}.pickle"
    if os.path.exists(picklefile):
        ((c_pos, c_neg), (o_pos, o_neg)) = pickle.load(open(picklefile, 'rb'))
    else:
        tinit=time()
        pair_cls.fit(Xtr, ytr)
        time_tr=time()-tinit

        c_pos, c_neg, _ = verification(pair_cls, Xte, yte, close_coordinates)
        o_pos, o_neg, time_te = verification(pair_cls, Xte_out, yte_out, open_coordinates)
        pickle.dump(((c_pos, c_neg), (o_pos, o_neg)), open(picklefile, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        csv.write(f'{opt.dataset}\t{method}\t{time_tr}\t{time_te}\n')

    plot(c_pos, c_neg, method, path=f'../plot-sav/{opt.dataset}/{method}{learner_st}_close.png', decision=0.5)
    plot(o_pos, o_neg, method, path=f'../plot-sav/{opt.dataset}/{method}{learner_st}_open.png', decision=0.5)


def plot(positive, negative, method, path, logscale=False, decision=None):
    sns.set_context("talk")
    parent_dir = Path(path).parent
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    if logscale:
        print(np.min(positive), np.min(negative))
        epsilon = .01
        positive = np.log(np.asarray(positive) + epsilon)
        negative = np.log(np.asarray(negative) + epsilon)
        if decision is not None:
            decision = np.log(decision+epsilon)


    # seaborn histogram
    sns.distplot(positive, hist=True, kde=True, bins=20, color='blue', hist_kws={'edgecolor': 'black'}, norm_hist=True, label='positive')
    sns.distplot(negative, hist=True, kde=True, bins=20, color='green', hist_kws={'edgecolor': 'black'}, norm_hist=True, label='negative')
    if decision is not None:
        plt.axvline(decision, label='threshold', linestyle='--', color='k', linewidth=3)


    # Add labels
    plt.title(method)
    if logscale:
        plt.xlabel('log(score)')
    else:
        plt.xlabel('score')
    plt.ylabel('frequency')
    plt.legend()
    plt.gcf().subplots_adjust(bottom=0.15, right=.98, top=.92)
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    # plt.show()


def verification(cls, Xte, yte, coordinates, n_jobs=-1):
    tinit=time()

    slices = get_parallel_slices(n_tasks=len(coordinates), n_jobs=n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(task_sav)(
            cls,
            coordinates[slice_i],
            Xte, yte, job
        )
        for job, slice_i in enumerate(slices)
    )
    pos,neg = list(zip(*results))
    pos = list(itertools.chain.from_iterable(pos))
    neg = list(itertools.chain.from_iterable(neg))
    tend = time()-tinit

    return pos, neg, tend


def verification_via_impostors(impostors, X, y, joblist_coordinates, n_jobs=-1):
    tinit=time()
    precomputed_impostors = impostors.precompute_impostors(X, n_jobs=n_jobs)
    slices = get_parallel_slices(n_tasks=len(joblist_coordinates), n_jobs=n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(task_sav_imp)(
            impostors,
            joblist_coordinates[slice_i],
            X, y, precomputed_impostors, job
        )
        for job, slice_i in enumerate(slices)
    )

    pos,neg = list(zip(*results))
    pos = list(itertools.chain.from_iterable(pos))
    neg = list(itertools.chain.from_iterable(neg))
    tend = time()-tinit
    return pos, neg, tend


def task_sav(cls, coordinates, X, y, job):
    total = 0
    correct = 0
    positive, negative = [], []
    pbar = tqdm(coordinates)
    for i, j in pbar:
        same_pred = cls.h_prob(X[i], X[j])
        same_truth = (y[i] == y[j])
        if same_truth:
            positive.append(same_pred)
        else:
            negative.append(same_pred)
        total += 1
        pbar.set_description(f'sav job={job} (acc={100*correct/total:.2f}%)')
    return positive, negative


def task_sav_imp(impostors, coordinates, X, y, precomputed_impostors, job):
    total = 0
    correct = 0
    positive, negative = [], []
    pbar = tqdm(coordinates)
    for i, j in pbar:
        same_pred = impostors.same_author_score(X[i], X[j], xi_impostors=precomputed_impostors[i], xj_impostors=precomputed_impostors[j])
        same_truth = (y[i] == y[j])
        if same_truth:
            positive.append(same_pred)
        else:
            negative.append(same_pred)
        total += 1
        pbar.set_description(f'sav impostors job={job} (acc={100*correct/total:.2f}%)')
    return positive, negative


if __name__ == '__main__':

    available_datasets = {'imdb62', 'pan2011', 'victorian', 'arxiv'}
    available_learners = {'LR', 'SVM'}

    # Training settings
    parser = argparse.ArgumentParser(description='This method performs experiments regarding the classification-by-pairs')
    parser.add_argument('dataset', type=str, metavar='DATASET', help=f'Name of the dataset to run experiments on')
    parser.add_argument('learner', type=str, metavar='LEARNER', help=f'Base learner (valid ones are {available_learners})')
    parser.add_argument('--n_authors', type=int, default=10, metavar='N', help='Number of authors to extract')
    parser.add_argument('--n_open_authors', type=int, default=10, metavar='N', help='Number of authors for open set SAV')
    parser.add_argument('--docs_by_author', type=int, default=50, metavar='N', help='Number of texts by author to extract')
    parser.add_argument('--pickle', type=str, default=None, help='path to a file where to save/load a pickle of the dataset once built')
    parser.add_argument('--logfile', type=str, default='../log-sav/results.csv', help='File csv with results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--rawfreq', default=False, action='store_true', help='use raw frequencies instead of tfidf')

    opt = parser.parse_args()

    assert_opt_in(opt.dataset, 'dataset', available_datasets)
    assert_opt_in(opt.learner, 'learner', available_learners)

    main()
