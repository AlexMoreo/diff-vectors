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


class LinearCombinationRule:
    def __init__(self, learner, diff_function, standardize=True):
        self.standardizer=StandardizeTransformer() if standardize else None
        self.diff=diff_function
        self.cls_combination = learner

    def fit(self, X, y):
        self.Xref=X
        Z = pairwise_diff(self.diff, X)
        if self.standardizer is not None:
            Z = self.standardizer.fit_transform(Z)
        return self.cls_combination.fit(Z, y)

    def predict(self, X):
        Z = pairwise_diff(self.diff, X, self.Xref)
        if self.standardizer is not None:
            Z = self.standardizer.transform(Z)
        return self.cls_combination.predict(Z)


class ProbKNNCombinationRule:
    def __init__(self, diff_function, k):
        self.diff = diff_function
        self.k=k # -1 indicates self-tunning

    def fit(self, X, y):
        self.Xref=X
        self.yref=y

        if self.k == -1: # self tune the k parameter
            Z = pairwise_diff(self.diff, X)
            np.fill_diagonal(Z, 0) # do not take into account self-comparisons

            self.authors, self.docs_of_author = list(zip(*Counter(y).most_common()))
            self.n_authors=len(self.authors)
            self.max_docs_of_any_author = self.docs_of_author[0]

            A = self.doc_author_doc_matrix(Z)

            best_acc=0
            best_k=0
            for k in range(1,self.max_docs_of_any_author+1):
                S = A[:,:,:k].mean(axis=-1)
                pred_k = np.argmax(S, axis=-1)
                acc_k = (self.yref == pred_k).mean()
                print(f'k={k}, acc={acc_k:.3f} (bestk={best_k}, bestacc={best_acc:.3f})')
                if acc_k>best_acc:
                    best_acc=acc_k
                    best_k=k
            self.k=best_k

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
        Z = pairwise_diff(self.diff, X, self.Xref)
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
        assert isinstance(sav_classifier, PairSAVClassifier), 'wrong type for sav_classifier'
        assert cls_policy in PairAAClassifier.COMBINATION_POLICIES, f'unexpected policy {cls_policy}'
        assert cls_policy!='linear' or learner is not None, 'learner must be specified for cls_policy="linear"'
        self.combination_policy = cls_policy
        self.k = k

        self.cls = sav_classifier
        if self.combination_policy == 'linear':
            self.cls_combination = LinearCombinationRule(clone(learner), self.cls.h_prob, standardize=False)
        elif self.combination_policy == 'knn':
            self.cls_combination = ProbKNNCombinationRule(self.cls.h_prob, self.k)

    def fit(self, X, y):
        self.cls.calibrate(X,y)
        self.n_pairs_=self.cls.n_pairs_
        self.cls_combination.fit(X, y)
        return self

    def predict(self, X):
        return self.cls_combination.predict(X)

    @classmethod
    def diff_vectors(cls, Xa, Xb):
        # return Xa.multiply(Xb)
        #return (Xa - Xb).power(2., dtype=float)
        return abs(Xa-Xb)


class PairSAVClassifier(BaseEstimator):

    def __init__(self, learner, pos, neg, max=-1):
        self.pos_request = pos
        self.neg_request = neg
        self.max_request = max
        self.pos_selection_policy='round_robin'
        self.isfit=False
        self.cls=learner

    def h_logits(self, xi, xj):
        return self.cls.decision_function(PairSAVClassifier.diff_vectors(xi, xj))[0]

    def h_prob(self, xi, xj):
        proba = self.cls.predict_proba(PairSAVClassifier.diff_vectors(xi, xj))[0, np.argwhere(self.cls.classes_ == 1)]
        return proba.flatten()[0]

    def same_author(self, xi, xj):
        return self.h_prob(xi, xj) > 0.5

    def fit(self, X, y):
        pg = PairGenerator(y, pos_request=self.pos_request, neg_request=self.neg_request, max_request=self.max_request,
                           pos_selection_policy=self.pos_selection_policy)
        Xa = X[pg.pairs[:, 0]]
        Xb = X[pg.pairs[:, 1]]
        XPtr = PairSAVClassifier.diff_vectors(Xa, Xb)
        yPtr = pg.labels
        self.n_pairs_ = len(pg.pairs)
        self.cls.fit(XPtr, yPtr)
        self.isfit = True
        self.XPtr, self.yPtr = XPtr, yPtr

        return self

    def calibrate(self, X, y):
        if not self.isfit:
            self.fit(X, y)
        if not hasattr(self.cls, 'predict_proba'):
            print(f'the classifier is not probabilistic... calibrating')
            if isinstance(self.cls, GridSearchCV):
                self.cls = self.cls.best_estimator_
            self.cls = CalibratedClassifierCV(self.cls, cv=5, ensemble=False).fit(self.XPtr, self.yPtr)

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

    def fit(self, X, y):
        pg = PairGenerator(y, pos_request=self.pos_request, neg_request=self.neg_request, max_request=self.max_request)
        Xa = X[pg.pairs[:, 0]]
        Xb = X[pg.pairs[:, 1]]

        d = paired_distances(Xa, Xb, metric=self.metric, n_jobs=self.n_jobs).reshape(-1,1)
        self.dist_range = (d.min(),d.max())
        y = pg.labels
        self.n_pairs_ = len(pg.pairs)
        self.cls.fit(d, y)
        return self

    def h_prob(self, xi, xj):
        d = paired_distances(xi, xj, metric=self.metric, n_jobs=self.n_jobs).reshape(-1,1)
        proba = self.cls.predict_proba(d)[0, np.argwhere(self.cls.classes_ == 1)]
        return proba.flatten()[0]



