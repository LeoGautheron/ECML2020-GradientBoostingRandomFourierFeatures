#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from sklearn.svm import SVR
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class BMKR(object):
    def __init__(self, T=100, C=1):
        self.T = T
        self.C = C
        self.WeakLearners = []
        self.parametersKernels = []
        for i in range(-4, 6):
            self.parametersKernels.append({"kernel": "rbf", "gamma": 2**i})
        self.parametersKernels.append({"kernel": "linear", "gamma": 1})

    def fit(self, X, y):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(X.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        yTilde = y
        for t in range(self.T):
            clfs = []
            mses = []
            for p in self.parametersKernels:
                clf = SVR(kernel=p["kernel"],
                          gamma=p["gamma"],
                          C=self.C,
                          max_iter=1e4)
                clf.fit(X, yTilde)
                yTildePred = clf.predict(X)
                mses.append(0.5*np.sum((yTildePred-yTilde)**2))
                clfs.append(clf)
            idxBest = np.argmin(mses)
            clf = clfs[idxBest]
            yTilde -= clf.predict(X)
            self.WeakLearners.append(clf)

    def predict(self, X):
        yTildePreds = [self.WeakLearners[i].predict(X)
                       for i in range(self.T)]
        pred = np.sum(yTildePreds, axis=0)
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred
