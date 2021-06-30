from sklearn.base import BaseEstimator
import numpy as np
from scipy.sparse import csr_matrix
from joblib import Parallel, delayed
import itertools
from sklearn.metrics import f1_score
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from utils.common import get_parallel_slices, get_verification_coordinates


def minmax(u,v):
    den=csr_matrix.maximum(u,v).sum()
    if den!=0:
        num=csr_matrix.minimum(u,v).sum()
        return num / den
    else:
        return 0


def cosine(u,v):
    return cosine_similarity(u,v)[0,0]


class PairImpostors(BaseEstimator):
    """
    The Impostors method for Authorship Verification described in:
    Koppel, Moshe, and Yaron Winter. "Determining if two documents are written by the same author."
    Journal of the Association for Information Science and Technology 65.1 (2014): 178-187.
    """

    def __init__(self, sigma=-1., top_impostors=250, n_impostors=25, k=100, similarity=minmax):
        """
        Koppel's "Impostors" method for same-author verification
        :param sigma: the condifende threshold (to be optimized in a validation set)
        :param top_impostors: maximum number of impostors to generate (m=250 in the paper)
        :param n_impostors: number of randomly chosen impostors from the top_impostors (n=10 in the paper)
        :param k: number of iterations for the randomly chosen feature set checks (k=100 in the paper)
        :param similarity: a callable function that computes the similarity between two np.ndarray (similarity=minmax in the paper)
        """
        self.sigma = sigma
        self.m=top_impostors
        self.n=n_impostors
        self.k=k
        self.sim=similarity

    def fit(self, X, y):
        nF = X.shape[1]
        self.random_feature_choices = np.vstack([np.random.choice(nF,nF//2,replace=False) for _ in range(self.k)])
        self.Xref = X
        return self

    def generate_impostors(self, x):
        similarities = np.array([self.sim(x,xi) for xi in self.Xref])
        most_similars = np.argsort(similarities)[-self.m:]
        return self.Xref[most_similars]

    def score(self, x, v, v_impostors):
        # computes the number of choices of features sets (out of k) for which sim(x,v) > sim(x, imp_v_i) for all i
        more_similar_count = 0
        for feature_choice in self.random_feature_choices:
            xf = x[0,feature_choice]
            vf = v[0,feature_choice]
            sim_xv = self.sim(xf, vf)

            rand_index = np.random.choice(range(v_impostors.shape[0]), self.n)
            v_rand_impostors = v_impostors[rand_index]

            most_similar = True
            v_impostors_feat = v_rand_impostors[:,feature_choice]
            for impostor in v_impostors_feat:
                sim_xvi = self.sim(xf, impostor)
                if sim_xv < sim_xvi:
                    most_similar = False
                    break
            if most_similar:
                more_similar_count += 1

        return more_similar_count / self.k

    def same_author_score(self, xi, xj, xi_impostors=None, xj_impostors=None):
        if xj_impostors is None:
            xj_impostors = self.generate_impostors(xj)
        if xi_impostors is None:
            xi_impostors = self.generate_impostors(xi)

        score_xi = self.score(xi, xj, xj_impostors)
        score_xj = self.score(xj, xi, xi_impostors)

        ave_score = (score_xi+score_xj)/2

        return ave_score

    def same_author(self, xi, xj, xi_impostors=None, xj_impostors=None):
        return self.same_author_score(xi, xj, xi_impostors, xj_impostors) > self.sigma

    def precompute_impostors(self, X, n_jobs=-1):
        nD = X.shape[0]

        # precompute the impostors (in parallel)
        slices = get_parallel_slices(n_tasks=nD, n_jobs=n_jobs)
        precomputed_impostors = Parallel(n_jobs=n_jobs)(
            delayed(_impostors_task_generate_impostors)(
                self, X[slice_i], job
            )
            for job, slice_i in enumerate(slices)
        )
        return list(itertools.chain.from_iterable(precomputed_impostors))


def optimize_sigma(X, y, top_impostors=250, n_impostors=25, k=100, similarity=minmax, balanced=True, subsample=500):
    classes = np.unique(y)
    nC = len(classes)
    nD = X.shape[0]
    tr_y, te_y = set(classes[:nC//2]), set(classes[nC//2:])
    tr_index = np.array([i for i in range(nD) if y[i] in tr_y])
    te_index = np.array([i for i in range(nD) if y[i] in te_y])
    Xtr, Ytr = X[tr_index], y[tr_index]
    Xte, Yte = X[te_index], y[te_index]

    impostors_method = PairImpostors(-1, top_impostors, n_impostors, k, similarity)
    impostors_method.fit(Xtr, Ytr)
    coordinates = get_verification_coordinates(Yte, balanced=balanced, subsample=500)
    scores, labels_same = generate_scores(impostors_method, Xte, Yte, coordinates)

    # seek for the best sigma
    sort_index = np.argsort(scores)
    scores = scores[sort_index]
    labels_same = labels_same[sort_index]
    n_scores = len(scores)

    best_f1 = None
    best_sigma = None
    pbar = tqdm(list(range(1,n_scores-1)))
    for i in pbar:
        if scores[i] == 0: continue
        y_pred = [0]*i + [1]*(n_scores-i)
        f1 = f1_score(labels_same, y_pred, average='binary')
        if best_f1 is None or f1 > best_f1:
            best_f1=f1
            best_sigma = scores[i]
            pbar.set_description(f'best f1={f1:.3f} for sigma={scores[i]} in position {i}/{n_scores}')

    return best_sigma


def generate_scores(impostors, X, y, joblist_coordinates, n_jobs=-1):

    precomputed_impostors = impostors.precompute_impostors(X, n_jobs=n_jobs)

    # generate all scores of similarity between pairs of documents in joblist_coordinates (in parallel)
    slices = get_parallel_slices(n_tasks=len(joblist_coordinates), n_jobs=n_jobs)
    results = Parallel(n_jobs=n_jobs)(
        delayed(_impostors_task_sav_scores)(
            impostors,
            joblist_coordinates[slice_i],
            X, y, precomputed_impostors, job
        )
        for job, slice_i in enumerate(slices)
    )
    scores, true_labels = list(zip(*results))
    scores = np.array(list(itertools.chain.from_iterable(scores)))
    true_labels = np.array(list(itertools.chain.from_iterable(true_labels)))

    return scores, true_labels


def _impostors_task_generate_impostors(impostors, X, job):
    pbar = tqdm(list(X))
    precomputed_impostors = []
    for xi in pbar:
        xi_impostors = impostors.generate_impostors(xi)
        precomputed_impostors.append(xi_impostors)
        pbar.set_description(f'precomputing impostors job={job}')
    return precomputed_impostors


def _impostors_task_sav_scores(impostors, coordinates, X, y, precomputed_impostors, job):
    total = 0
    scores, true_labels = [], []
    pbar = tqdm(coordinates)
    for i, j in pbar:
        same_score = impostors.same_author_score(X[i], X[j], xi_impostors=precomputed_impostors[i], xj_impostors=precomputed_impostors[j])
        same_truth = (y[i] == y[j])
        scores.append(same_score)
        true_labels.append(same_truth)
        total += 1
        pbar.set_description(f'sav impostors job={job}')
    return scores, true_labels


def _impostors_task_sav(impostors, coordinates, X, y, precomputed_impostors, job):
    total = 0
    correct = 0
    pred, truth = [], []
    pbar = tqdm(coordinates)
    for i, j in pbar:
        same_pred = impostors.same_author(X[i], X[j], xi_impostors=precomputed_impostors[i], xj_impostors=precomputed_impostors[j])
        same_truth = (y[i] == y[j])
        if same_pred == same_truth:
            correct += 1
        pred.append(same_pred)
        truth.append(same_truth)
        total += 1
        f1 = f1_score(truth, pred)
        pbar.set_description(f'sav impostors job={job} (acc={100*correct/total:.2f}% f1={100*f1:.3f}%)')
    return correct, total, pred, truth

