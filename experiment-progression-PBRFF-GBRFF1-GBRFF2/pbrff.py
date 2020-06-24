#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from scipy.special import logsumexp
from sklearn.svm import SVC


class pbrff(object):
    def __init__(self, beta=1, K=100, gamma=1, pctLandmarks=None,
                 n_landmarks=None, C=1, randomState=None):
        self.pctLandmarks = pctLandmarks
        self.n_landmarks = n_landmarks
        self.C = C
        self.randomState = randomState

        self.K = K
        self.beta = beta
        self.gamma = gamma

    def select_landmarks(self, X, y):
        if self.n_landmarks is None:
            self.n_landmarks = int(len(y)*self.pctLandmarks)
        idxsLandmaks = self.randomState.choice(len(X), self.n_landmarks,
                                               replace=True)
        self.landmarks_X = X[idxsLandmaks, :]
        self.landmarks_y = y[idxsLandmaks]

    def transform_cos(self, omega, delta):
        return np.cos(np.dot(delta, omega))

    def fit(self, X, y):
        self.select_landmarks(X, y)

        self.n, self.d = X.shape

        # Compute loss for a given number of Fourier features per landmarks.
        # Randomly sampling Omega from the Fourier distribution
        self.Omega = self.randomState.randn(self.n_landmarks,
                                            self.d, self.K) * (
                                                             2*self.gamma)**0.5
        loss = []
        # Computing loss for each landmarks
        for i in range(self.n_landmarks):
            transformed_X = self.transform_cos(self.Omega[i],
                                               X - self.landmarks_X[i])
            lambda_y = -np.ones(self.n)
            lambda_y[y == self.landmarks_y[i]] = 1

            landmark_loss = lambda_y @ transformed_X

            # For the random method, case where X_i == landmark needs to be
            # substracted
            landmark_loss = (landmark_loss - 1) / (self.n - 1)

            landmark_loss = (1 - landmark_loss) / 2
            loss.append(landmark_loss)
        loss = np.array(loss)

        # Compute pseudo-posterior Q distribution over the Fourier features.
        # Computing t
        to = self.beta * self.n**0.5
        # Computing Q
        self.Q = -to*loss - logsumexp(-to*loss, axis=1).reshape(-1, 1)
        self.Q = np.exp(self.Q)

        self.clf = SVC(kernel="linear", C=self.C, max_iter=1e4,
                       random_state=np.random.RandomState(1))
        self.clf.fit(self.transform(X), y)

    def transform(self, X):
        mapped_X = []
        for i in range(self.n_landmarks):
            transformed_X = self.transform_cos(self.Omega[i],
                                               X - self.landmarks_X[i])
            mapped_X.append(np.sum(transformed_X * self.Q[i], 1))
        return np.array(mapped_X).T

    def predict(self, X):
        return self.clf.predict(self.transform(X))
