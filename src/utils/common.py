from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np
import scipy
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
from joblib import Parallel, delayed
import multiprocessing
import itertools
from scipy.stats import ttest_ind_from_stats
from utils.lightweight_random_indexing import SVMLRI


def feature_selection(X, y, Xte, num_feats):
    feature_selector = SelectKBest(chi2, k=num_feats)
    X = feature_selector.fit_transform(X, y)
    Xte = feature_selector.transform(Xte)
    return X, Xte


class StandardizeTransformer:

    def __init__(self, axis=0):
        self.axis = axis
        self.yetfit = False
        self.scaler = StandardScaler()

    def fit(self, X):
        self.scaler.fit(X)
        self.yetfit=True
        return self

    def transform(self, X):
        if not self.yetfit: 'transform called before fit'
        return self.scaler.transform(X)

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def zscores(x, axis=0): #scipy.stats.zscores does not avoid division by 0, which can indeed occur
    std = np.clip(np.std(x, axis=axis, ddof=1), 1e-5, None)
    mean = np.mean(x, axis=axis)
    return (x - mean) / std


"""
projects elements in X1 into the space induced by applying h to the elements of X2. I.e., the resulting matrix
P contains elements pij=h(X1[i],X2[j]) and shape=(X1.shape[0], X2.shape[0])
"""
def pairwise_diff_(diff, X1, X2=None):
    return pairwise_distances(X1, X2, metric=diff, n_jobs=-1) # <-- this implementation is much slower


"""
projects elements in X1 into the space induced by applying h to the elements of X2. I.e., the resulting matrix
P contains elements pij=h(X1[i],X2[j]) and shape=(X1.shape[0], X2.shape[0])
"""
def pairwise_diff__(diff, X1, X2=None):
    h = diff
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False
    n = X1.shape[0]
    m = X2.shape[0]
    P = np.zeros(shape=(n, m))
    for i, xi in tqdm(enumerate(X1), 'projecting into the h space'):
        for j, xj in enumerate(X2):
            if j >= i:
                P[i, j] = h(xi, xj)
            else:
                if symmetric:
                    P[i, j] = P[j, i]
                else:
                    P[i, j] = h(xi, xj)
    return P


def pairwise_diff(diff, X1, X2=None):
    h = diff
    if X2 is None:
        X2 = X1
        symmetric = True
    else:
        symmetric = False
    n = X1.shape[0]
    m = X2.shape[0]

    joblist_coordinates = [(i, j) for i in range(n) for j in range(i if symmetric else 0,m)]

    slices = get_parallel_slices(n_tasks=len(joblist_coordinates), n_jobs=-1)

    diffs = Parallel(n_jobs=-1)(
        delayed(pairwise_diff_task)(
            h, X1, X2, joblist_coordinates[slice_i], job
        )
        for job,slice_i in enumerate(slices)
    )
    diffs = list(itertools.chain.from_iterable(diffs))

    P = np.zeros(shape=(n, m))
    for (i,j), diff_i in zip(joblist_coordinates,diffs):
        P[i,j] = diff_i

    if symmetric:
        for i in range(1,n):
            for j in range(i):
                P[i,j]=P[j,i]

    return P


def pairwise_diff_task(h, X1, X2, coordinates, job):
    return [h(X1[i], X2[j]) for i,j in tqdm(coordinates, desc=f'projecting onto the h space (job={job})')]


# given the number of tasks and the number of jobs, generates the slices for the parallel threads
def get_parallel_slices(n_tasks, n_jobs=-1):
    if n_jobs==-1:
        n_jobs = multiprocessing.cpu_count()
    batch = int(n_tasks / n_jobs)
    remainder = n_tasks % n_jobs
    return [slice(job*batch, (job+1)*batch+ (remainder if job == n_jobs - 1 else 0)) for job in range(n_jobs)]


# given a target vector containing the true classes generates a list of index coordinates for all distinct pairs
# to test for verification (if balanced=True then all positive pairs are generated and an equal number of negative pairs
# is randomly sampled from all negatives -- which are comparatively many more)
# if subsample is specified (by setting the subsample arg), then a (balanced--if requested) subsample is returned
def get_verification_coordinates(y, balanced=True, subsample=-1, shuffle=True):
    nD = len(y)
    x_coordinates = np.array([(i, j) for i in range(nD) for j in range(i + 1, nD)])
    if balanced:
        y_coordinates = np.array([y[i] == y[j] for i, j in x_coordinates])
        n_positives = y_coordinates.sum()
        n_negatives = len(y_coordinates) - n_positives
        x_coordinates = np.concatenate([
            x_coordinates[y_coordinates==1], # all positives
            x_coordinates[y_coordinates==0][np.random.choice(n_negatives, size=n_positives, replace=False)] # a random subsample
        ])
    if subsample != -1 and subsample < len(x_coordinates):
        y_coordinates = np.array([y[i] == y[j] for i, j in x_coordinates])
        total = len(y_coordinates)
        n_positives = y_coordinates.sum()
        n_negatives = total - n_positives
        take_pos = int((n_positives/total)*subsample)
        take_neg = subsample - take_pos
        x_coordinates = np.concatenate([
            x_coordinates[y_coordinates == 1][np.random.choice(n_positives, size=take_pos, replace=False)],
            x_coordinates[y_coordinates == 0][np.random.choice(n_negatives, size=take_neg, replace=False)]
            # a random subsample
        ])
    if shuffle:
        np.random.shuffle(x_coordinates)
    return x_coordinates


# gets a random sample of X,y extracting cat_size documents from each distinct category in y
def random_sample(X, y, cat_size):
    nD = X.shape[0]
    assert nD == len(y), 'wrong size'
    classes = np.unique(y)
    X_,y_=[],[]
    for c in classes:
        Xsel = X[y==c]
        indexes = np.random.permutation(Xsel.shape[0])[:cat_size]
        Xsel = Xsel[indexes]
        ysel = np.full(Xsel.shape[0], c)
        X_.append(Xsel)
        y_.append(ysel)
    X_ = scipy.sparse.vstack(X_)
    y_ = np.concatenate(y_)
    return X_, y_

def prepare_learner(learner):
    Cs = np.logspace(-3, 3, 7)
    if learner == 'LR':
        print('returning LR')
        learner = LogisticRegressionCV(Cs=Cs, n_jobs=-1, cv=5)
        # learner = LogisticRegression()
        max = 50000
    elif learner == 'SVM':
        learner = CalibratedClassifierCV(SVMLRI(k=1000), cv=3, n_jobs=-1)
        max = 10000
    elif learner == 'SGD':
        learner = SGDClassifier(n_jobs=-1, loss='modified_huber')
        max = 500000
    elif learner == 'MNB':
        learner = MultinomialNB(alpha=0, force_alpha=True)
        max = 50000
    else:
        raise ValueError('unknown learner method')
    return learner, max


def ttest(mean1,std1, count1, mean2, std2, count2):
    _,pvalues = ttest_ind_from_stats(mean1, std1, count1, mean2, std2, count2)
    def symbol(pvalue):
        if pvalue==1:
            return '='
        if pvalue > 0.05:
            return '**'
        elif pvalue > 0.005:
            return '*'
        return ''
    return np.array([symbol(pval) for pval in pvalues])

