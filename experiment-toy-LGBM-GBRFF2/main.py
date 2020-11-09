#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gzip
import math
import pickle
import random
import sys

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from lightgbm import LGBMClassifier
from gbrff2 import gbrff as gbrff2

n_samples = np.arange(50, 2001, 50)
datasets = ["swiss", "circles", "board"]
nbIter = 1000
meshPrecision = 0.2
nbFoldValid = 5
seed = 1
if len(sys.argv) == 2:
    seed = int(sys.argv[1])


def listP(dic, shuffle=False):
    """
    Input: dictionnary with parameterName: array parameterRange
    Output: list of dictionnary with parameterName: parameterValue
    """
    # Recover the list of parameter names.
    params = list(dic.keys())
    # Initialy, the list of parameter to use is the list of values of
    # the first parameter.
    listParam = [{params[0]: value} for value in dic[params[0]]]
    # For each parameter p after the first, the "listParam" contains a
    # number x of dictionnary. p can take y possible values.
    # For each value of p, create x parameter by adding the value of p in the
    # dictionnary. After processing parameter p, our "listParam" is of size x*y
    for i in range(1, len(params)):
        newListParam = []
        currentParamName = params[i]
        currentParamRange = dic[currentParamName]
        for previousParam in listParam:
            for value in currentParamRange:
                newParam = previousParam.copy()
                newParam[currentParamName] = value
                newListParam.append(newParam)
        listParam = newListParam.copy()
    if shuffle:
        random.shuffle(listParam)
    return listParam


def make_board(n):
    n_cols = 4
    n_rows = 4
    Xx = np.random.uniform(low=0, high=n_cols, size=(n))
    Xy = np.random.uniform(low=0, high=n_rows, size=(n))
    X = np.array([Xx, Xy]).T
    y = []
    for x in X:
        row = math.floor(x[0])
        col = math.floor(x[1])
        if (row+col) % 2 == 0:
            y.append(-1)
        else:
            y.append(1)
    y = np.array(y)
    return X, y


def make_circles(n_samples, n_circles=4, noise=0.07, factor=1):
    mult = 1
    label = -1
    Xx = []
    Xy = []
    y = []
    for i in range(n_circles):
        n_samples_circle = n_samples // n_circles
        linspace = np.linspace(0, 2 * np.pi, n_samples_circle, endpoint=False)
        Xx.extend((np.cos(linspace) + np.random.normal(
                                   scale=noise, size=n_samples_circle)) * mult)
        Xy.extend((np.sin(linspace) + np.random.normal(
                                   scale=noise, size=n_samples_circle)) * mult)
        for j in range(n_samples_circle):
            y.append(label)
        mult += factor
        label *= -1
    X = np.array([Xx, Xy]).T
    y = np.array(y)
    return X, y


def make_swiss_roll(n):
    nbRoll = 1
    noise = 0.40
    nPerRoll = n//2
    t = 1.5 * np.pi * (1 + 2 * np.random.uniform(low=0, high=nbRoll,
                                                 size=nPerRoll))
    Xx1 = t * np.cos(t)
    Xy1 = t * np.sin(t)
    X1 = np.array([Xx1, Xy1]).T
    X1 += noise * np.random.randn(nPerRoll, 2)
    t = 1.5 * np.pi * (1 + 2 * np.random.uniform(low=0, high=nbRoll,
                                                 size=nPerRoll))
    Xx2 = 0.9*t * np.cos(t)
    Xy2 = 0.9*t * np.sin(t)
    X2 = np.array([Xx2, Xy2]).T
    X2 += noise * np.random.randn(nPerRoll, 2)
    X = np.vstack((X1, X2))
    y = np.array([-1]*nPerRoll + [1]*nPerRoll)
    return X, y


def applyAlgo(algo, p, Xtrain, Ytrain, Xtest, Ytest):
    if algo == "LGBM":
        clf = LGBMClassifier(n_estimators=nbIter, max_depth=p["max_depth"],
                             num_leaves=2**p["max_depth"],
                             reg_lambda=p["reg_lambda"], n_jobs=1,
                             random_state=1, verbose=-1)
    elif algo == "GBRFF2":
        clf = gbrff2(Lambda=p["Lambda"], gamma=p["gamma"]/Xtrain.shape[1],
                     T=nbIter, randomState=np.random.RandomState(1))
    clf.fit(Xtrain, Ytrain)
    perf = []
    perfIter = {}
    Ytrain_pred = clf.predict(Xtrain)
    Ytest_pred = clf.predict(Xtest)
    for true, pred, name in [(Ytrain, Ytrain_pred, "train"),
                             (Ytest, Ytest_pred, "test")]:
        # Compute performance measures by comparing prediction with true
        # labels
        tn, fp, fn, tp = confusion_matrix(true, pred, labels=[-1, 1]).ravel()
        perfIter[name] = ((int(tn), int(fp), int(fn), int(tp)))
    perf.append(perfIter)
    return perf, clf


listParams = {}
parLGBM = listP({"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                 "reg_lambda": [0] + [2**i for i in [-5, -4, -3, -2]]})
parGBRFF2 = listP({"Lambda": [0] + [2**i for i in [-5, -4, -3, -2]],
                   "gamma": [2**i for i in [-2, -1, 0, 1, 2]]})
listParams["LGBM"] = parLGBM
listParams["GBRFF2"] = parGBRFF2
listNames = {a: [] for a in listParams.keys()}
listParametersNames = {a: {} for a in listParams.keys()}
for a in listParams.keys():
    for i, p in enumerate(listParams[a]):
        listParametersNames[a][str(p)] = p
        listNames[a].append(str(p))
algos = list(listParams.keys())
r = {}
for ns, n in enumerate(n_samples):
    r[n] = {}
    for dataset in datasets:
        np.random.seed(seed)
        random.seed(seed)
        if dataset == "swiss":
            X, y = make_swiss_roll(n)
        elif dataset == "circles":
            X, y = make_circles(n)
        elif dataset == "board":
            X, y = make_board(n)
        Xtrain, Xtest, ytrain, ytest = train_test_split(
                                 X, y, shuffle=True, stratify=y, test_size=0.5)
        skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
        foldsTrainValid = list(skf.split(Xtrain, ytrain))
        xMin, xMax = min(X[:, 0]), max(X[:, 0])
        yMin, yMax = min(X[:, 1]), max(X[:, 1])
        xx, yy = np.meshgrid(np.arange(xMin, xMax+meshPrecision,
                                       meshPrecision),
                             np.arange(yMin, yMax+meshPrecision,
                                       meshPrecision))
        mesh = np.c_[xx.ravel(), yy.ravel()]
        r[n][dataset] = {"Xtrain": Xtrain, "ytrain": ytrain, "Xtest": Xtest,
                         "ytest": ytest, "xMin": xMin, "xMax": xMax,
                         "yMin": yMin, "yMax": yMax, "xx": xx, "yy": yy,
                         "mesh": mesh, "algos": {}}
        rValid = {a: {} for a in listParametersNames.keys()}
        for a in listParametersNames.keys():  # For each algo
            valids = []
            for nameP in listNames[a]:  # For each set of parameters
                p = listParametersNames[a][nameP]
                rValid[a][nameP] = []
                # Compute performance on each validation fold
                for iFoldVal in range(nbFoldValid):
                    fTrain, fValid = foldsTrainValid[iFoldVal]
                    perf, clf = applyAlgo(a, p,
                                          Xtrain[fTrain], ytrain[fTrain],
                                          Xtrain[fValid], ytrain[fValid])
                    tn, fp, fn, tp = perf[-1]["test"]
                    rValid[a][nameP].append((tp+tn)/(tp+tn+fp+fn))
                validAccuracy = np.mean(rValid[a][nameP])*100
                valids.append((validAccuracy, nameP))
                print(n, dataset, a,
                      "valid Accuracy {:5.2f}".format(validAccuracy), p)
            best = sorted(valids)[-1]
            best = (best[0], listParametersNames[a][best[1]])
            print("Best valid {:5.2f}".format(best[0]), "with", best[1])
            perf, clf = applyAlgo(a, best[1], Xtrain, ytrain, Xtest, ytest)
            tn, fp, fn, tp = perf[-1]["test"]
            testAccuracy = 100*(tp+tn)/(tp+tn+fp+fn)
            tn, fp, fn, tp = perf[-1]["train"]
            trainAccuracy = 100*(tp+tn)/(tp+tn+fp+fn)
            print(n, dataset, a,
                  "train Accuracy {:5.2f}".format(trainAccuracy),
                  "test Accuracy {:5.2f}".format(testAccuracy))
            if a.startswith("LGBM"):
                probas = clf.predict_proba(mesh)
                highest = np.argmax(probas, axis=1)
                Z = []
                for i, high in enumerate(highest):
                    if high == 0:
                        Z.append(-probas[i][high])
                    else:
                        Z.append(probas[i][high])
                Z = np.array(Z).reshape(xx.shape)
            else:
                Z = clf.decision_function(mesh).reshape(xx.shape)
            r[n][dataset][a] = {"trainAccuracy": trainAccuracy,
                                "testAccuracy": testAccuracy,
                                "Z": Z}
        f = gzip.open("res" + str(seed) + ".pklz", "wb")
        pickle.dump({"res": r, "algos": list(listParametersNames.keys()),
                     "datasets": datasets, "n_samples": n_samples[:ns+1]}, f)
        f.close()
