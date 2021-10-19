from utils.calibration import CalibratedClassifierCV
from utils.common import StandardizeTransformer, pairwise_diff
from collections import Counter
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC, LinearSVC
from sklearn.base import clone
import numpy as np
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from model.pair_generator import PairGenerator
from sklearn.metrics.pairwise import paired_distances
import scipy


class LinearCombinationRule:
    def __init__(self, learner, diff_function, standardize=True):
        self.standardize = standardize
        self.diff_function = diff_function
        self.learner = learner

    def fit(self, X, y):
        self.standardizer = StandardizeTransformer() if self.standardize else None
        self.Xref = X
        Z = pairwise_diff(self.diff_function, X)
        if self.standardizer is not None:
            Z = self.standardizer.fit_transform(Z)
        return self.learner.fit(Z, y)

    def predict(self, X):
        Z = pairwise_diff(self.diff_function, X, self.Xref)
        if self.standardizer is not None:
            Z = self.standardizer.transform(Z)
        return self.learner.predict(Z)

    def decision_function(self, X):
        Z = pairwise_diff(self.diff_function, X, self.Xref)
        if self.standardizer is not None:
            Z = self.standardizer.transform(Z)
        return self.learner.decision_function(Z)


class ProbKNNCombinationRule:
    def __init__(self, diff_function, k):
        self.diff_function = diff_function
        self.k = k  # -1 indicates self-tunning

    def fit(self, X, y):
        self.Xref = X
        self.yref = y

        if self.k == -1:  # self tune the k parameter
            Z = pairwise_diff(self.diff_function, X)
            np.fill_diagonal(Z, 0)  # do not take into account self-comparisons

            self.authors, self.docs_of_author = list(zip(*Counter(y).most_common()))
            self.n_authors=len(self.authors)
            self.max_docs_of_any_author = self.docs_of_author[0]

            A = self.doc_author_doc_matrix(Z)

            best_acc=0
            best_k=0
            for k in range(1, self.max_docs_of_any_author+1):
                S = A[:,:,:k].mean(axis=-1)
                pred_k = np.argmax(S, axis=-1)
                acc_k = (self.yref == pred_k).mean()
                print(f'k={k}, acc={acc_k:.3f} (bestk={best_k}, bestacc={best_acc:.3f})')
                if acc_k>best_acc:
                    best_acc=acc_k
                    best_k=k
                if best_k == 1.0:
                    break
            self.k = best_k

        return self

    def doc_author_doc_matrix(self, Z):
        n = Z.shape[0]
        A = np.zeros(shape=(n, self.n_authors, self.max_docs_of_any_author))
        for author, n_docs_of_author in zip(self.authors, self.docs_of_author):
            author_posteriors = Z[:, self.yref == author]
            author_posteriors[:, ::-1].sort()
            A[:, author, :n_docs_of_author] = author_posteriors
            author_posterior_expected = author_posteriors.mean(axis=-1).reshape(-1,1)
            if n_docs_of_author<self.max_docs_of_any_author:
                A[:, author, n_docs_of_author:] = author_posterior_expected  # fill the remaining cells with the average (so that it does not modify the average when averaged...)
        return A

    def decision_function(self, X):
        Z = pairwise_diff(self.diff_function, X, self.Xref)
        A = self.doc_author_doc_matrix(Z)
        scores = A[:, :, :self.k].mean(axis=-1)
        return scores

    def predict(self, X):
        S = self.decision_function(X)
        prediction = np.argmax(S, axis=-1)
        return prediction


class PairAAClassifier(BaseEstimator):
    # different policies to determine the authorship based on the classified test pairs
    COMBINATION_POLICIES = ['linear', 'knn']

    def __init__(self, sav_classifier, cls_policy, k=-1, learner=None):
        self.sav_classifier = sav_classifier
        self.cls_policy = cls_policy
        self.k = k
        self.learner = learner

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self

    def fit(self, X, y):
        print('fit')
        assert isinstance(self.sav_classifier, PairSAVClassifier), 'wrong type for sav_classifier'
        assert self.cls_policy in PairAAClassifier.COMBINATION_POLICIES, f'unexpected policy {self.cls_policy}'
        assert self.cls_policy != 'linear' or self.learner is not None, 'learner must be specified for cls_policy="linear"'

        if self.cls_policy == 'linear':
            self.cls_combination = LinearCombinationRule(clone(self.learner), self.sav_classifier.h_prob, standardize=False)
        elif self.cls_policy == 'knn':
            self.cls_combination = ProbKNNCombinationRule(self.sav_classifier.h_prob, self.k)

        self.sav_classifier.calibrate(X,y)
        self.n_pairs_ = self.sav_classifier.n_pairs_
        self.cls_combination.fit(X, y)
        print('fit-end')
        return self

    def predict(self, X):
        return self.cls_combination.predict(X)

    # def decision_function(self, X):
    #     return self.cls_combination.decision_function(X)

    def predict_proba(self, X):
        decisions = self.cls_combination.predict(X)
        return np.vstack([1-decisions, decisions]).T

    @classmethod
    def diff_vectors(cls, Xa, Xb):
        # return Xa.multiply(Xb)
        #return (Xa - Xb).power(2., dtype=float)
        return abs(Xa-Xb)


def generate_pairs_helper(X, y, pos_request, neg_request, max_request, aux_Xys=None, verification_task=False):
    Xs, ys = [X], [y]
    if aux_Xys is not None:
        for Xi, yi in aux_Xys:
            Xs.append(Xi)
            ys.append(yi)

    Xas, Xbs, yPtrs = [], [], []
    lens = 0
    for Xi, yi in zip(Xs, ys):
        pg = PairGenerator(yi, pos_request=pos_request, neg_request=neg_request, max_request=max_request,
                           verification_task=verification_task)
        Xas.append(Xi[pg.pairs[:, 0]])
        Xbs.append(Xi[pg.pairs[:, 1]])
        yPtrs.append(pg.labels)
        lens += len(pg.pairs)
        print('added ', len(pg.pairs))
    print('total pairs ', lens)

    Xa = scipy.sparse.vstack(Xas)
    Xb = scipy.sparse.vstack(Xbs)
    yPtr = np.concatenate(yPtrs)

    return Xa, Xb, yPtr, lens


class PairSAVClassifier(BaseEstimator):

    def __init__(self, learner, pos_request, neg_request, max_request=-1, verification_task=False):
        """
        A classifier that learns the concept SAME vs DIFFERENT from diff-vector representations (representations of
        vector differences between elements from the same or different group)
        Parameters
        ----------
        learner The underlying binary classifier
        pos_request The maximum number of positive (SAME) elements to generate; set to -1 for generating them all
        neg_request The maximum number of negative (DIFFERENT) elements to generate; set to -1 for generating them all
        max_request The maximum number of elements to generate (both SAME + DIFFERENT)
        verification_task Set to True whenever the task is binary and the class 0 represents the complement of class 1,
            i.e., when the class 0 represents documents from authors other than the author 1, but not necessarily from
            a single individual (this is important in order to avoid creating pairs of "SAME" from the negative class).
        """
        self.pos_request = pos_request
        self.neg_request = neg_request
        self.max_request = max_request
        self.isfit = False
        self.learner = learner
        self.verification_task = verification_task

    def set_params(self, **params):
        if not params:
            return self

        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.kwargs[key] = value

        return self

    def h_logits(self, xi, xj):
        return self.learner.decision_function(PairSAVClassifier.diff_vectors(xi, xj))[0]

    def h_prob(self, xi, xj):
        proba = self.learner.predict_proba(PairSAVClassifier.diff_vectors(xi, xj))[0, np.argwhere(self.learner.classes_ == 1)]
        return proba.flatten()[0]

    def same_author(self, xi, xj):
        return self.h_prob(xi, xj) > 0.5

    def fit(self, X, y, aux_Xys=None):
        print('generating pairs')
        Xa, Xb, yPtr, lenpairs = generate_pairs_helper(X, y, self.pos_request, self.neg_request, self.max_request,
                                                       aux_Xys, self.verification_task)

        print(f'generating diff vectors Xa={Xa.shape} Xb={Xb.shape}')
        XPtr = PairSAVClassifier.diff_vectors(Xa, Xb)
        self.n_pairs_ = lenpairs
        print('fitting learner within PairSAVClassifier')
        self.learner.fit(XPtr, yPtr)
        print('fitting learner within PairSAVClassifier [DONE]')
        self.isfit = True
        self.XPtr, self.yPtr = XPtr, yPtr

        return self

    def calibrate(self, X, y):
        if not self.isfit:
            self.fit(X, y)
        if not hasattr(self.learner, 'predict_proba'):
            print(f'the classifier is not probabilistic... calibrating')
            if isinstance(self.learner, GridSearchCV):
                self.learner = self.learner.best_estimator_
            self.learner = CalibratedClassifierCV(self.learner, cv=5, ensemble=False).fit(self.XPtr, self.yPtr)

    @classmethod
    def diff_vectors(cls, Xa, Xb):
        # return Xa.multiply(Xb)
        #return (Xa - Xb).power(2., dtype=float)
        return abs(Xa-Xb)


class DistanceSAVClassifier:

    def __init__(self, pos, neg, max = -1, metric='euclidean', n_jobs=-1):
        self.pos_request = pos
        self.neg_request = neg
        self.max_request = max
        self.metric = metric
        self.n_jobs = n_jobs
        self.cls = LogisticRegression(C=1)

    def same_author(self, xi, xj):
        d = paired_distances(xi, xj, metric=self.metric, n_jobs=self.n_jobs).reshape(-1,1)
        return self.cls.predict(d)

    def fit(self, X, y, aux_Xys=None):
        Xa, Xb, yPtr, lenpairs = generate_pairs_helper(X, y, self.pos_request, self.neg_request, self.max_request, aux_Xys)

        d = paired_distances(Xa, Xb, metric=self.metric, n_jobs=self.n_jobs).reshape(-1,1)
        self.dist_range = (d.min(),d.max())
        self.n_pairs_ = lenpairs
        # print('ZEROS:0', yPtr[d.flatten()==0].mean())
        self.cls.fit(d, yPtr)
        return self

    def h_prob(self, xi, xj):
        d = paired_distances(xi, xj, metric=self.metric, n_jobs=self.n_jobs).reshape(-1,1)
        proba = self.cls.predict_proba(d)[0, np.argwhere(self.cls.classes_ == 1)]
        return proba.flatten()[0]



