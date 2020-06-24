# Code adapted from the original implementation from
# https://bitbucket.org/doglic/gfc

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from sklearn.svm import SVC


def fitt(X, y, gamma):
    K = X.T.dot(X) / X.shape[0]
    K.flat[::K.shape[0] + 1] += gamma ** 2 + 1e-10
    amplitudes = np.linalg.solve(K, X.T.dot(y) / X.shape[0])
    return amplitudes, K


class GreedyFeatureConstruction():
    def __init__(self, numFeatures=100, C=1, maxFminIters=200,
                 randomState=np.random):
        self.numFeatures = numFeatures
        self.C = C
        self.maxFminIters = maxFminIters
        self.randomState = randomState

    def mse(self, hparams, X, y, bias=None):
        F, z = self._transform(X, hparams[:-1], bias)
        alpha, K = fitt(F, y, hparams[-1])
        residues = F.dot(alpha) - y
        loss = np.mean(residues**2)
        gradient = np.zeros(hparams.shape[0])
        sin_2z_tr, cos_2z_tr = np.sin(2 * z), np.cos(2 * z)
        alpha, K = fitt(F, y, hparams[-1])
        t = np.linalg.solve(K, np.mean(np.multiply(residues.reshape(-1, 1), F), axis=0))
        gradient[-1] -= 2 * hparams[-1] * t.dot(alpha)
        gradient[:-1] += np.mean(np.multiply(np.multiply(alpha[1] * F[:, 2] - alpha[2] * F[:, 1],
                                                        residues).reshape(-1, 1), X), axis=0)
        v = np.multiply(y, t[1] * F[:, 2] - t[2] * F[:, 1]).reshape(-1, 1)
        v -= (t[0] * alpha[1] + t[1] * alpha[0]) * np.multiply(F[:, 0], F[:, 2]).reshape(-1, 1)
        v += (t[0] * alpha[2] + t[2] * alpha[0]) * np.multiply(F[:, 0], F[:, 1]).reshape(-1, 1)
        v -= (t[1] * alpha[2] + t[2] * alpha[1]) * cos_2z_tr.reshape(-1, 1)
        v -= (t[1] * alpha[1] - t[2] * alpha[2]) * sin_2z_tr.reshape(-1, 1)
        gradient[:-1] += np.mean(np.multiply(v, X), axis=0)
        return loss, 2 * gradient

    def _transform(self, X, spectra, bias=None):
        if len(spectra.shape) == 1:
            spectra = spectra.reshape(-1, 1)
        Z = X.dot(spectra)
        n, m = Z.shape
        start_column = 0
        if bias is not None:
            representation_dim = 2 * m + 1
        else:
            representation_dim = 2 * m
        F = np.zeros((n, representation_dim))
        if bias is not None:
            F[:, 0] = bias
            start_column += 1
        sin_cols = np.arange(start_column, F.shape[1], 2, dtype=int)
        cos_cols = sin_cols + 1
        F[:, sin_cols] = np.sin(Z)
        F[:, cos_cols] = np.cos(Z)
        return F, Z

    def transform(self, X):
        return self._transform(X, self.W.T)[0]

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.empty((self.numFeatures, d))
        F = np.zeros((X.shape[0], 3))
        F[:, 0] = np.mean(y)
        for i in range(self.numFeatures):
            spectrum = self.randomState.standard_normal(size=d) / np.sqrt(d)
            init_hparams = np.append(spectrum, 1e-2)
            fmin_output = fmin_l_bfgs_b(func=lambda z: self.mse(z, X, y, F[:, 0]), x0=init_hparams,
                                        disp=False, maxiter=self.maxFminIters)
            self.W[i, :] = fmin_output[0][:-1]
            F[:, 1:] = self._transform(X, self.W[i, :])[0]
            amplitudes = fitt(F, y, fmin_output[0][-1])[0]
            F[:, 0] = F.dot(amplitudes)
        self.clf = SVC(kernel="linear", C=self.C, max_iter=1e4,
                       random_state=np.random.RandomState(1))
        self.clf.fit(self.transform(X), y)

    def predict(self, X):
        return self.clf.predict(self.transform(X))
