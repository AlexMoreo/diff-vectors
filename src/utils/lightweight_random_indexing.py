import math
import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.base import BaseEstimator
from sklearn.svm import LinearSVC


class RandomIndexingBoC(object):

    def __init__(self, latent_dimensions, non_zeros=2, seed=None):
        self.latent_dimensions = latent_dimensions
        self.k = non_zeros
        self.ri_dict = None
        self.seed = seed

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def fit(self, X):
        nF = X.shape[1]
        nL = self.latent_dimensions
        format = 'csr' if issparse(X) else 'np'
        self.ri_dict = _create_random_index_dictionary(shape=(nF, nL),
                                                       k=self.k, normalized=True, format=format, seed=self.seed)
        return self

    def transform(self, X):
        assert X.shape[1] == self.ri_dict.shape[0], 'feature space is inconsistent with the RI dictionary'
        if self.ri_dict is None:
            raise ValueError("Error: transform method called before fit.")
        P = X.dot(self.ri_dict)
        if issparse(P):
            P.sort_indices()
        return P


def _create_random_index_dictionary(shape, k, normalized=False, format='csr', positive=False, seed=None):
    assert format in ['csr', 'np'], 'Format should be in "[csr, np]"'
    nF, latent_dimensions = shape
    print("Creating the random index dictionary for |V|={} with {} dimensions".format(nF,latent_dimensions))
    val = 1.0 if not normalized else 1.0/math.sqrt(k)
    ri_dict = np.zeros((nF, latent_dimensions))
    if seed is not None:
        np.random.seed(seed)

    #TODO: optimize
    for t in range(nF):
        dims = np.zeros(k, dtype=np.int32)
        dims[0] = t % latent_dimensions #the first dimension is choosen in a round-robin manner (prevents gaps)
        dims[1:] = np.random.choice(latent_dimensions, size=k-1, replace=False)
        values = (np.random.randint(0,2, size=k)*2.0-1.0) * val if not positive else np.array([+val]*k)
        ri_dict[t,dims]=values
        print("\rprogress [%.2f%% complete]" % (t * 100.0 / nF), end='')
    print('\nDone')

    if format=='csr':
        ri_dict = csr_matrix(ri_dict)

    return ri_dict


class SVMLRI(LinearSVC):
    
    def __init__(self, k, C=1):
        super().__init__(C=C)
        self.k = k

    def fit(self, X, y, sample_weight=None):
        # self.svm = LinearSVC(C=self.C)
        self.lri = RandomIndexingBoC(self.k, seed=0)
        print(f'fitting LRI with k={self.k}')
        X = self.lri.fit_transform(X)
        print(f'X has now shape {X.shape}')
        return super().fit(X, y, sample_weight=sample_weight)
        # return self.svm.fit(X, y, sample_weight=sample_weight)
    
    def predict(self, X):
        X = self.lri.transform(X)
        return super().predict(X)
        # return self.svm.predict(X)

    def decision_function(self, X):
        X = self.lri.transform(X)
        return super().decision_function(X)
        # return self.svm.decision_function(X)

    # @property
    # def classes_(self):
    #     return self.svm.classes_

