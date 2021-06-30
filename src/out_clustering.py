from data.fetch_enron_mail import EnronMail
from data.fetch_pan2011 import Pan2011
from data.fetch_imdb62 import Imdb62
from utils.common import pairwise_diff, remove_label_gaps
from utils.result_manager import Result
from sklearn.model_selection import train_test_split
from feature_extraction.author_vectorizer import FeatureExtractor
from model.pair_classification import PairAAClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import LinearSVC
import random
from utils.evaluation import *
import os
import argparse
from time import time
import pickle
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.cluster import OPTICS, SpectralClustering,KMeans, AgglomerativeClustering
from sklearn import metrics


def main():

    random.seed(7)
    np.random.seed(7)

    Cs = [1,10,100,1000]

    # dataset fetching
    if opt.dataset == 'enron':
        dataset = EnronMail(n_authors=opt.n_authors, docs_by_author=opt.docs_by_author)
        Xtr, Xte, ytr, yte = train_test_split(dataset.data, dataset.labels, test_size=0.25, random_state=42, stratify=dataset.labels)
    elif opt.dataset == 'Imdb_62':
        dataset = Imdb62(n_authors=opt.n_authors, docs_by_author=opt.docs_by_author, n_open_set_authors=opt.n_open_authors)

        # Xtr, Xte, ytr, yte = train_test_split(dataset.data, dataset.labels, test_size=0.25, random_state=42, stratify=dataset.labels)
    elif opt.dataset == 'pan2011':
        path = "../data"
        dataset = Pan2011(path, 'small', n_authors=opt.n_authors, docs_by_author=opt.docs_by_author, min_length=-1)
        Xtr, ytr = dataset.train_data, dataset.train_labels
        Xte, yte = dataset.test_data, dataset.test_labels

    Xtr, ytr = dataset.train_data, dataset.train_labels
    Xte, yte = dataset.test_data, dataset.test_labels
    ytr, yte = remove_label_gaps(ytr, yte)

    # feature extraction
    te_init = time()
    vectorizer = FeatureExtractor('english', cleaning=False)
    Xtr = vectorizer.fit_transform(Xtr, ytr)
    Xte = vectorizer.transform(Xte, None)
    vectorizer_time = time() - te_init

    print(f'Xtr={Xtr.shape}')
    print(f'Xte={Xte.shape}')

    clusters_raw = AgglomerativeClustering(n_clusters=opt.n_open_authors, linkage='complete').fit_predict(Xte.toarray())
    for cluster in np.unique(clusters_raw):
        print(f'\tcluster {cluster}: labels={yte[clusters_raw==cluster]}')

    pos = -1  # -1 stands for all
    neg = -1  # -1 stands for the same number as pos
    max = 50000
    # if opt.method == 'PairCLSknn':
    #     if opt.k != -1: opt.method+=f'-k{opt.k}'
    cls = PairAAClassifier(pos, neg, max=max, cls_policy='knn', k=-1, Cs=Cs)
    cls.fit(Xtr, ytr)

    Zte = 1-pairwise_diff(diff=cls.h_prob, X1=Xte)


    # clusters = OPTICS(min_samples=1, n_jobs=-1, metric=cls.h_prob).fit_predict(Xte.toarray())
    # clusters = SpectralClustering(n_clusters=n_test_authors, n_jobs=-1).fit_predict(Xte.toarray())
    # clusters_diff = AgglomerativeClustering(n_clusters=n_test_authors, affinity='precomputed', linkage='complete').fit_predict(Zte)
    clusters_diff = AgglomerativeClustering(distance_threshold=0.5, n_clusters=None, affinity='precomputed', linkage='complete').fit_predict(Zte)

    for cluster in np.unique(clusters_diff):
        print(f'\tcluster {cluster}: labels={yte[clusters_diff==cluster]}')

    # evaluation
    raw_ARI = metrics.adjusted_rand_score(yte, clusters_raw)
    diff_ARI = metrics.adjusted_rand_score(yte, clusters_diff)

    raw_AMI = metrics.adjusted_mutual_info_score(yte, clusters_raw)
    diff_AMI = metrics.adjusted_mutual_info_score(yte, clusters_diff)

    raw_HOM = metrics.homogeneity_score(yte, clusters_raw)
    diff_HOM = metrics.homogeneity_score(yte, clusters_diff)

    raw_COM = metrics.completeness_score(yte, clusters_raw)
    diff_COM = metrics.completeness_score(yte, clusters_diff)

    raw_FOW = metrics.fowlkes_mallows_score(yte, clusters_raw)
    diff_FOW = metrics.fowlkes_mallows_score(yte, clusters_diff)

    print(f'ARS raw={raw_ARI:.3} diff={diff_ARI:.3f}')
    print(f'AMI raw={raw_AMI:.3} diff={diff_AMI:.3f}')
    print(f'HOM raw={raw_HOM:.3} diff={diff_HOM:.3f}')
    print(f'COM raw={raw_COM:.3} diff={diff_COM:.3f}')
    print(f'FOW raw={raw_FOW:.3} diff={diff_FOW:.3f}')







if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='This method performs experiments regarding the classification-by-pairs')
    parser.add_argument('--dataset', type=str, default='Imdb_62', metavar='STR',
                        help=f'Name of the dataset to run experiments on')
    parser.add_argument('--n_authors', type=int, default=5, metavar='N', help='Number of authors to extract')
    parser.add_argument('--n_open_authors', type=int, default=10, metavar='N', help='Number of authors to extract')
    parser.add_argument('--docs_by_author', type=int, default=10, metavar='N', help='Number of texts by author to extract')
    parser.add_argument('--k', type=int, default=-1, metavar='N', help='k for knn prob mean [-1, indicates self tunning]')
    parser.add_argument('--logfile', type=str, default='../log/results.csv', help='File csv with results')
    parser.add_argument('--method', type=str, default='PairCLS', help='Classification method')

    opt = parser.parse_args()

    assert opt.dataset in {'Imdb_62', 'pan2015', 'pan2011', 'enron'}, f'unknown pretrained set {opt.pretrained}'

    main()
