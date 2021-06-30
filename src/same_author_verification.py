import importlib
import random
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import precision_recall_fscore_support
from main import prepare_dataset, assert_opt_in, prepare_learner
from model.pair_impostors import PairImpostors, minmax, optimize_sigma, cosine, _impostors_task_sav
from utils.common import get_verification_coordinates, get_parallel_slices, random_sample
from utils.result_manager import SAVResult, check_if_already_performed, count_results
from feature_extraction.author_vectorizer import FeatureExtractor
from model.pair_classification import PairAAClassifier, DistanceSAVClassifier, PairSAVClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
from utils.evaluation import *
import os
import argparse
from tqdm import tqdm
import pickle
from sklearn.base import clone
from joblib import Parallel, delayed
import itertools


def prepare_dataset(dataset, n_authors, docs_by_author, n_open_set_authors, seed=42, picklepath=None, rawfreq=False):

    if picklepath and os.path.exists(picklepath):
        print('pre-built dataset exists, loading...')
        (Xtr,ytr,Xte,yte,Xte_out,yte_out) = pickle.load(open(picklepath, 'rb'))
    else:
        print('building dataset...')
        dataset_loader = getattr(importlib.import_module(f'data.fetch_{dataset.lower()}'), dataset.title())
        dataset = dataset_loader(
            n_authors=n_authors, docs_by_author=docs_by_author, n_open_set_authors=n_open_set_authors, random_state=seed
        )

        Xtr, ytr = dataset.train.data, dataset.train.target
        Xte, yte = dataset.test.data, dataset.test.target
        Xte_out, yte_out = dataset.test_out.data, dataset.test_out.target

        # feature extraction
        print(f'using raw_freq = {rawfreq}')
        vectorizer = FeatureExtractor('english', cleaning=False, use_raw_frequencies=rawfreq,
                                      function_words=True,
                                      word_lengths=True,
                                      sentence_lengths=True,
                                      punctuation=True,
                                      post_ngrams=True,
                                      word_ngrams=True,
                                      char_ngrams=True)
        Xtr = vectorizer.fit_transform(Xtr, ytr)
        Xte = vectorizer.transform(Xte, None)
        if dataset.test_out is not None:
            Xte_out = vectorizer.transform(Xte_out, None)

        if picklepath:
            print('pickling built dataset...')
            pickle.dump((Xtr, ytr, Xte, yte, Xte_out, yte_out), open(picklepath, 'wb'), pickle.HIGHEST_PROTOCOL)

    return Xtr, ytr, Xte, yte, Xte_out, yte_out


def main():

    csv = SAVResult(opt.logfile)

    # dataset preparation
    n_authors = opt.n_authors
    n_open_authors = opt.n_open_authors
    docs_by_author = opt.docs_by_author
    docs_by_author_te = opt.docs_by_author
    Xtr, ytr, Xte, yte, Xte_out, yte_out = prepare_dataset(
        opt.dataset,
        n_authors=n_authors, docs_by_author=docs_by_author, n_open_set_authors=n_open_authors,
        seed=opt.seed, picklepath=opt.pickle, rawfreq=opt.rawfreq
    )

    #reset the random seed (of the dataset is generated then it uses random functions, if it is loaded, then not)
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    # classifier instantiation
    base_learner = prepare_learner(Cs = [1, 10, 100, 1000], learner=opt.learner)

    max = 15000 // 2 if opt.learner == 'SVM' else 50000

    Xte, yte = random_sample(Xte, yte, cat_size=docs_by_author_te)
    Xte_out, yte_out = random_sample(Xte_out, yte_out, cat_size=docs_by_author_te)
    min_docs_by_author = 5 # cannot perform grid search with less

    close_coordinates = get_verification_coordinates(yte, balanced=True, subsample=1000)
    open_coordinates = get_verification_coordinates(yte_out, balanced=True, subsample=1000)

    n, k = 10, 100 # parameters set according to the original Koppel's article
    # m = min(250, Xtr.shape[0] // 10)
    m = 50 # m = 250 in Koppel's article, but they use many more potential candidates
    similarity = cosine
    #sigma = optimize_sigma(Xtr, ytr, top_impostors=m, n_impostors=n, k=k, similarity=similarity)

    while docs_by_author==-1 or docs_by_author >= min_docs_by_author:

        pair_cls = PairSAVClassifier(clone(base_learner), -1, -1, max=max)
        base_cls = clone(base_learner)
        dist_l1_cls = DistanceSAVClassifier(-1, -1, max=max, metric='manhattan')
        dist_l2_cls = DistanceSAVClassifier(-1, -1, max=max, metric='euclidean')
        dist_cos_cls = DistanceSAVClassifier(-1, -1, max=max, metric='cosine')
        nDtr = Xtr.shape[0]
        if m > nDtr//2:
            m = nDtr//2
            n = min(10, m // 2)
        #impostors_cls = PairImpostors(sigma=sigma, top_impostors=m, n_impostors=n, k=k, similarity=similarity)

        print(f'n_authors={n_authors} open_set_authors={n_open_authors}: docs_by_author={docs_by_author}')
        print(f'Xtr.shape={Xtr.shape} with clases={len(np.unique(ytr))}')
        print(f'Xte.shape={Xte.shape} with clases={len(np.unique(yte))}')
        print(f'Xte_out.shape={Xte_out.shape} with clases={len(np.unique(yte_out))}')

        method = 'Dist-l1-close'
        if not check_if_already_performed(opt.logfile, opt.dataset, opt.seed, n_authors, docs_by_author, method):
            print('training distance-based baseline')
            dist_l1_cls.fit(Xtr, ytr)
            dist_l2_cls.fit(Xtr, ytr)
            dist_cos_cls.fit(Xtr, ytr)
            csv.append(opt.dataset, opt.seed, 'CLOSE', n_authors, docs_by_author, n_authors, -1, 'Dist-l1-close', *verification(dist_l1_cls, Xte, yte, close_coordinates))
            csv.append(opt.dataset, opt.seed, 'CLOSE', n_authors, docs_by_author, n_authors, -1, 'Dist-l2-close', *verification(dist_l2_cls, Xte, yte, close_coordinates))
            csv.append(opt.dataset, opt.seed, 'CLOSE', n_authors, docs_by_author, n_authors, -1, 'Dist-cos-close', *verification(dist_cos_cls, Xte, yte, close_coordinates))
            csv.append(opt.dataset, opt.seed, 'OPEN', n_authors, docs_by_author, n_authors, -1, 'Dist-l1-open', *verification(dist_l1_cls, Xte_out, yte_out, open_coordinates))
            csv.append(opt.dataset, opt.seed, 'OPEN', n_authors, docs_by_author, n_authors, -1, 'Dist-l2-open', *verification(dist_l2_cls, Xte_out, yte_out, open_coordinates))
            csv.append(opt.dataset, opt.seed, 'OPEN', n_authors, docs_by_author, n_authors, -1, 'Dist-cos-open', *verification(dist_cos_cls, Xte_out, yte_out, open_coordinates))

        method=opt.learner + '-close'
        if not check_if_already_performed(opt.logfile, opt.dataset, opt.seed, n_authors, docs_by_author, method):
            print('training baseline')
            base_cls.fit(Xtr,ytr)
            print('testing baseline in SAV close set')
            r_base = verification_via_attribution(base_cls, Xte, yte, close_coordinates)
            csv.append(opt.dataset, opt.seed, 'CLOSE', n_authors, docs_by_author, n_authors, -1, method, *r_base)

        method = 'Pairknn-close'
        if not check_if_already_performed(opt.logfile, opt.dataset, opt.seed, n_authors, docs_by_author, method):
            print('training pair cls (knn)')
            pair_cls.fit(Xtr, ytr)
            print('testing pair cls (knn) in SAV close set')
            r_pair_close = verification(pair_cls, Xte, yte, close_coordinates)
            csv.append(opt.dataset, opt.seed, 'CLOSE', n_authors, docs_by_author, n_authors, -1, opt.learner + method, *r_pair_close)

            print('testing pair cls (knn) in SAV-via-attribution close set')
            r_pair_att_open = verification_via_attribution(pair_cls, Xte, yte, close_coordinates)
            csv.append(opt.dataset, opt.seed, 'CLOSE', n_authors, docs_by_author, n_authors, -1, opt.learner + 'Pairknn-att-close', *r_pair_att_open)

            print('testing pair cls (knn) in SAV open set')
            r_pair_open = verification(pair_cls, Xte_out, yte_out, open_coordinates)
            csv.append(opt.dataset, opt.seed, 'OPEN', n_authors, docs_by_author, n_open_authors, docs_by_author_te, opt.learner + 'Pairknn-open', *r_pair_open)

        #method = 'Impostors-open'
        #if not check_if_already_performed(opt.logfile, opt.dataset, opt.seed, n_authors, docs_by_author, method):
        #    print('training impostors')
        #    impostors_cls.fit(Xtr, ytr)
        #    print('testing impostors in SAV open set')
        #    r_imp = verification_via_impostors(impostors_cls, Xte_out, yte_out, open_coordinates)
        #    csv.append(opt.dataset, opt.seed, 'OPEN', n_authors, docs_by_author, n_open_authors, docs_by_author_te, method, *r_imp, notes=f'sigma={sigma} m={m} n={n}')

        docs_by_author -= 10
        if docs_by_author>=min_docs_by_author:
            Xtr, ytr = random_sample(Xtr, ytr, cat_size=docs_by_author)
            print(f'sampling: Xtr={Xtr.shape} ytr={ytr.shape}')

    csv.close()


def verification(cls, Xte, yte, coordinates):
    correct = 0
    total = 0
    pred, truth = [], []
    pbar = tqdm(coordinates)
    for i,j in coordinates:
        same_pred = cls.same_author(Xte[i], Xte[j])
        same_truth = (yte[i]==yte[j])
        if same_pred == same_truth:
            correct += 1
        pred.append(same_pred)
        truth.append(same_truth)
        total += 1
        pbar.set_description(f'authorship linking correct={correct}/{total} ({100*correct/total:.2f}%)')

    acc = correct*100./total
    p,r,f1,_ = precision_recall_fscore_support(truth, pred, average='binary', pos_label=1)
    print(f'acc={acc:.3f}% p={p:.3f} r={r:.3f} f1={f1:.3f}')

    return acc, p, r, f1


def verification_via_attribution(cls, Xte, yte, coordinates):
    yte_ = cls.predict(Xte)

    truth = [yte[i] == yte[j] for i, j in coordinates]
    pred  = [yte_[i] == yte_[j] for i, j in coordinates]
    correct = sum([t == p for t,p in zip(truth,pred)])
    total = len(truth)

    acc = correct*100./total
    p,r,f1,_ = precision_recall_fscore_support(truth, pred, average='binary', pos_label=1)
    print(f'acc={acc:.3f}% p={p:.3f} r={r:.3f} f1={f1:.3f}')

    return acc, p, r, f1


def verification_trivial_rejector(yte, coordinates):
    truth = [yte[i] == yte[j] for i, j in coordinates]
    correct = len(truth) - sum(truth)
    total = len(truth)

    acc = correct*100./total
    p, r, f1 = 1, 0, 0
    print(f'acc={acc:.3f}% p={p:.3f} r={r:.3f} f1={f1:.3f}')

    return acc, p, r, f1


def verification_via_impostors(impostors, X, y, joblist_coordinates, n_jobs=-1):

    precomputed_impostors = impostors.precompute_impostors(X, n_jobs=n_jobs)

    # generate all scores of similarity between pairs of documents in joblist_coordinates (in parallel)
    slices = get_parallel_slices(n_tasks=len(joblist_coordinates), n_jobs=n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_impostors_task_sav)(
            impostors,
            joblist_coordinates[slice_i],
            X, y, precomputed_impostors, job
        )
        for job, slice_i in enumerate(slices)
    )
    correct,total,pred,truth = list(zip(*results))
    correct = sum(correct)
    total = sum(total)
    pred = list(itertools.chain.from_iterable(pred))
    truth = list(itertools.chain.from_iterable(truth))

    acc = correct * 100. / total
    p,r,f1,_ = precision_recall_fscore_support(truth, pred, average='binary', pos_label=1)
    print(f'acc={acc:.3f}% p={p:.3f} r={r:.3f} f1={f1:.3f}')

    return acc, p, r, f1


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

    if count_results(opt.logfile, opt.dataset, opt.seed, opt.n_authors) == 25:
        print('nothing to do here! all results were already computed. Exit')
        import sys
        sys.exit(0)

    main()
