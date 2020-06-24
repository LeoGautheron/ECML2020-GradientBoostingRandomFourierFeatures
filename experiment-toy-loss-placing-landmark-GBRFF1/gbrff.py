#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import fminbound


class gbrff(object):
    def __init__(self, gamma=1, T=200, learning_rate=0.1,
                 init="Mean", randomState=np.random):
        self.T = T
        self.randomState = randomState
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.init = init

    def predictWeak(self, X, xt, omega, alpha):
        return alpha*np.cos(np.dot(omega, (xt-X).T)).T

    def miLoss(self, n):
        self.xts[self.t][0] = 2*np.pi*n/self.omega[0]
        v0 = np.exp(-self.yTilde*np.cos(np.dot(self.omega,
                                               (self.xts[self.t]-self.X).T)))
        return (1/self.n)*np.sum(v0)

    def muLoss(self, xt):
        v0 = np.exp(-self.yTilde*np.cos(np.dot(self.omega, (xt-self.X).T)))
        return (1/self.n)*np.sum(v0)

    def fit(self, X, y, landmarksToTest=None, labels=None):
        # Assuming there is two labels in y. Convert them in -1 and 1 labels.
        if labels is None:
            labels = sorted(np.unique(y))
        self.negativeLabel, self.positiveLabel = labels[0], labels[1]
        newY = np.ones(X.shape[0])  # Set all labels at 1
        newY[y == labels[0]] = -1  # except the smallest label in y at -1.
        y = newY
        self.X = X
        self.n, self.d = self.X.shape
        if self.init == "Mean":
            self.initPred = np.mean(y)
        else:
            self.initPred = self.init
        curPred = np.array([float(self.initPred) for i in range(self.n)])
        w = np.exp(-y*curPred)
        self.yTilde = y*w
        self.omegas = self.randomState.randn(self.T, self.d)*(
                                                             2*self.gamma)**0.5
        self.alphas = np.ones(self.T)
        self.xts = np.zeros((self.T, self.d))
        self.debug = []
        for self.t in range(self.T):
            self.omega = self.omegas[self.t]
            bestN = fminbound(self.miLoss, 0, 1, xtol=1e-2)
            self.xts[self.t][0] = 2*np.pi*bestN/self.omega[0]
            if landmarksToTest is not None and self.t == self.T-1:
                losses = []
                for landmarkToTest in landmarksToTest:
                    losses.append(self.muLoss(landmarkToTest))
                self.debug.append(losses)
            yTildePred = self.predictWeak(self.X, self.xts[self.t], self.omega,
                                          1)
            numerator = np.einsum("n,n", 1+y*yTildePred, w)
            denominator = np.einsum("n,n", 1-y*yTildePred, w)
            self.alphas[self.t] = 0.5*np.log(numerator/denominator)
            curPred += self.learning_rate*self.alphas[self.t]*yTildePred
            w = np.exp(-y*curPred)
            self.yTilde = y*w

    def predict(self, X, numberWeakClassifiers=None):
        if numberWeakClassifiers is None:
            numberWeakClassifiers = len(self.alphas)
        yTildePreds = [self.predictWeak(X, self.xts[i], self.omegas[i],
                                        self.alphas[i])
                       for i in range(numberWeakClassifiers)]
        if len(yTildePreds) == 0:
            pred = np.sign([self.initPred]*len(X))
        else:
            pred = np.sign(self.initPred+self.learning_rate*np.sum(
                                                          yTildePreds, axis=0))
        # Then convert back the labels -1 and 1 to the labels given in fit
        yPred = np.array([self.positiveLabel] * X.shape[0])
        yPred[pred == -1] = self.negativeLabel
        return yPred

    def decision_function(self, X, numberWeakClassifiers=None):
        if numberWeakClassifiers is None:
            numberWeakClassifiers = len(self.alphas)
        yTildePreds = [self.predictWeak(X, self.xts[i], self.omegas[i],
                                        self.alphas[i])
                       for i in range(numberWeakClassifiers)]
        if len(yTildePreds) == 0:
            pred = np.array([self.initPred]*len(X))
        else:
            pred = self.initPred+self.learning_rate*np.sum(yTildePreds, axis=0)
        return pred
