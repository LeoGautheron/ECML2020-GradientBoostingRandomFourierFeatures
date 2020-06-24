#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fminbound


class gbrff(object):
    def __init__(self, gamma=0.1, T=100, randomState=np.random):
        self.T = T
        self.randomState = randomState
        self.gamma = gamma

    def fit(self, X, y):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(X.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        self.n, d = X.shape
        meanY = np.mean(y)
        self.initPred = 0.5*np.log((1+meanY)/(1-meanY))
        curPred = np.full(self.n, self.initPred)
        pi2 = np.pi*2
        self.omegas = self.randomState.randn(self.T, d)*(2*self.gamma)**0.5
        self.alphas = np.empty(self.T)
        self.xts = np.empty(self.T)
        self.X = X
        self.XT = X.T
        for t in range(self.T):
            wx = self.omegas[t].dot(self.XT)
            w = np.exp(-y*curPred)
            self.yTilde = y*w
            self.yTildeN = -self.yTilde
            self.xts[t] = pi2*fminbound(lambda n: np.sum(np.exp(
                       self.yTildeN*np.cos(pi2*n - wx))), -0.5, 0.5, xtol=1e-2)
            yTildePred = np.cos(self.xts[t] - wx)
            vi = (y*yTildePred).dot(w)
            vj = np.sum(w)
            alpha = 0.5*np.log((vj+vi)/(vj-vi))
            curPred += alpha*yTildePred
            self.alphas[t] = alpha

    def predict(self, X):
        pred = self.initPred+self.alphas.dot(
                                np.cos(self.xts[:, None]-self.omegas.dot(X.T)))
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred
