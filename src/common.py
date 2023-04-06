import numpy as np
from sklearn.linear_model import LogisticRegressionCV, SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC


def prepare_learner(learner):
    Cs = np.logspace(-3, 3, 7)
    if learner == 'LR':
        print('returning LR')
        learner = LogisticRegressionCV(Cs=Cs, n_jobs=-1, cv=5)
        # learner = LogisticRegression()
        max = 50000
    elif learner == 'SVM':
        #learner = GridSearchCV(estimator=SVC(kernel='linear', probability=True), param_grid={'C':Cs}, n_jobs=-1, cv=5)
        #learner = CalibratedClassifierCV(GridSearchCV(estimator=LinearSVC(), param_grid={'C': Cs}, n_jobs=-1, cv=5))
        #learner = CalibratedClassifierCV(LinearSVC(), cv=5)
        learner = GridSearchCV(estimator=LinearSVC(), param_grid={'C': Cs}, n_jobs=-1, cv=5)
        max = 15000 // 2
    elif learner == 'SGD':
        learner = SGDClassifier(n_jobs=-1, loss='modified_huber')
        max = 500000
    return learner, max



