#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import logsumexp


class gbrff(object):
    def __init__(self, gamma=1, T=100, beta=1, K=10,
                 randomState=np.random):
        self.T = T
        self.randomState = randomState
        self.gamma = gamma
        self.K = K
        self.beta = beta

    def fit(self, X, y):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(X.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        self.X = X
        self.n, d = self.X.shape
        meanY = np.mean(y)
        self.initPred = 0.5*np.log((1+meanY)/(1-meanY))
        curPred = np.full(self.n, self.initPred)
        self.omegas = self.randomState.randn(self.T, self.K, d)*(
                                                             2*self.gamma)**0.5
        self.alphas = np.empty(self.T)
        idxsLandmaks = self.randomState.choice(len(X), self.T, replace=True)
        self.xts = X[idxsLandmaks, :]
        self.Qls = np.empty((self.T, self.K))
        to = self.beta * self.n**0.5
        for t in range(self.T):
            self.omega = self.omegas[t]
            w = np.exp(-y*curPred)
            self.yTilde = y*w
            lossRandomFeatures = (1/self.n)*np.sum(np.exp(-self.yTilde*np.cos(
                               np.dot(self.omega, (self.xts[t]-X).T))), axis=1)
            self.Qls[t] = np.exp(-to*lossRandomFeatures - logsumexp(
                                                       -to*lossRandomFeatures))
            yTildePred = self.Qls[t].dot(np.cos(np.dot(self.omega, (
                                                            self.xts[t]-X).T)))
            vi = (y*yTildePred).dot(w)
            vj = np.sum(w)
            alpha = 0.5*np.log((vj+vi)/(vj-vi))
            curPred += alpha*yTildePred
            self.alphas[t] = alpha

    def predict(self, X):
        yTildePreds = [self.Qls[t].dot(np.cos(np.dot(self.omegas[t], (
                                                            self.xts[t]-X).T)))
                       for t in range(self.T)]
        pred = self.initPred+self.alphas.dot(yTildePreds)
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.full(X.shape[0], self.positiveLabel)
        yPred[pred < 0] = self.negativeLabel
        return yPred
