#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import os
import pickle
import random
import sys
import time

from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gbrff1 import gbrff as gbrff1

import datasets

###############################################################################
#                   Part of code about arguments to modify                    #
#                                                                             #

nbFoldValid = 5

minClass = +1
majClass = -1

#                                                                             #
#               End of part of code about arguments to modify                 #
###############################################################################
if not os.path.exists("results"):
    try:
        os.makedirs("results")
    except:
        pass


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


def applyAlgo(algo, p, Xtrain, Ytrain, Xtest, Ytest):
    scaler = StandardScaler()
    scaler.fit(Xtrain)
    Xtrain = scaler.transform(Xtrain)
    Xtest = scaler.transform(Xtest)

    if algo.startswith("GBRFF1;"):
        nbIter = int(algo.split(";")[1])
        K = int(algo.split(";")[2])
        clf = gbrff1(gamma=p["gamma"]/Xtrain.shape[1],
                     T=nbIter,
                     beta=p["beta"], K=K,
                     randomState=np.random.RandomState(1))

    clf.fit(Xtrain, Ytrain)
    perf = []
    perfIter = {}
    Ytrain_pred = clf.predict(Xtrain)
    Ytest_pred = clf.predict(Xtest)
    for true, pred, name in [(Ytrain, Ytrain_pred, "train"),
                             (Ytest, Ytest_pred, "test")]:
        # Compute performance measures by comparing prediction with true
        # labels
        tn, fp, fn, tp = confusion_matrix(true, pred,
                                          labels=[majClass,
                                                  minClass]).ravel()
        perfIter[name] = ((int(tn), int(fp), int(fn), int(tp)))
    perf.append(perfIter)
    return perf


###############################################################################
# Definition of parameters to test during the cross-validation for each algo
listParams = {}
parGBRFF1 = listP({"gamma": [2**i for i in [-2, -1, 0, 1, 2]],
                   "beta": np.logspace(-2, 2, 5)})
parGBRFF11 = listP({"gamma": [2**i for i in [-2, -1, 0, 1, 2]],
                    "beta": [1]})

for a in ["GBRFF1;" + str(i) + ";20" for i in [1, 5, 10, 15, 25, 50]]:
    listParams[a] = parGBRFF1

for a in ["GBRFF1;" + str(i) + ";10" for i in [2, 10, 20, 30, 50, 100]]:
    listParams[a] = parGBRFF1

for a in ["GBRFF1;" + str(i) + ";5" for i in [4, 20, 40, 60, 100, 200]]:
    listParams[a] = parGBRFF1

for a in ["GBRFF1;" + str(i) + ";1" for i in [20, 100, 200, 300, 500, 1000]]:
    listParams[a] = parGBRFF11

listNames = {a: [] for a in listParams.keys()}
listParametersNames = {a: {} for a in listParams.keys()}
for a in listParams.keys():
    for i, p in enumerate(listParams[a]):
        listParametersNames[a][str(p)] = p
        listNames[a].append(str(p))


r = {}  # All the results are stored in this dictionnary
datasetsDone = []
startTime = time.time()
for da in datasets.d.keys():  # For each dataset
    print(da)
    X, y = datasets.d[da][0], datasets.d[da][1]

    if len(sys.argv) == 2:
        np.random.seed(int(sys.argv[1]))
        random.seed(int(sys.argv[1]))
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, shuffle=True,
                                                    stratify=y, test_size=0.3)

    skf = StratifiedKFold(n_splits=nbFoldValid, shuffle=True)
    foldsTrainValid = list(skf.split(Xtrain, ytrain))

    r[da] = {"valid": {a: {} for a in listParametersNames.keys()},
             "test": {a: {} for a in listParametersNames.keys()},
             "time": {a: 0 for a in listParametersNames.keys()}}
    for a in listParametersNames.keys():  # For each algo
        nbParamToTest = len(listParametersNames[a])
        nbParamTested = 0
        startTime = time.time()
        for nameP in listNames[a]:  # For each set of parameters
            p = listParametersNames[a][nameP]
            r[da]["valid"][a][nameP] = []
            # Compute performance on each validation fold
            accsValid = []
            for iFoldVal in range(nbFoldValid):
                fTrain, fValid = foldsTrainValid[iFoldVal]
                t1 = time.time()
                perf = applyAlgo(a, p,
                                 Xtrain[fTrain], ytrain[fTrain],
                                 Xtrain[fValid], ytrain[fValid])
                r[da]["valid"][a][nameP].append(perf)
                tn, fp, fn, tp = perf[-1]["test"]
                accsValid.append((tp+tn)/(tp+tn+fp+fn))
            nbParamTested += 1
            # Compute performance on test set by training on the union of
            # all the validation folds.
            t1 = time.time()
            perf = applyAlgo(a, p, Xtrain, ytrain, Xtest, ytest)
            r[da]["test"][a][nameP] = perf
            tn, fp, fn, tp = perf[-1]["test"]
            acc = (tp+tn)/(tp+tn+fp+fn)
            tn, fp, fn, tp = perf[-1]["train"]
            accTrain = (tp+tn)/(tp+tn+fp+fn)
            print(da, a,
                  str(nbParamTested)+"/"+str(nbParamToTest),
                  "train Accuracy {:5.2f}".format(accTrain*100),
                  "valid Accuracy {:5.2f}".format(np.mean(accsValid)*100),
                  "test Accuracy {:5.2f}".format(acc*100), p)
        print(da, a, "time: {:8.2f} sec".format(time.time()-startTime))
        r[da]["time"][a] = time.time() - startTime
    datasetsDone.append(da)
    # Save the results at the end of each dataset
    if len(sys.argv) == 2:
        f = gzip.open("./results/resReal" + sys.argv[1] + ".pklz", "wb")
    else:
        f = gzip.open("./results/resReal" + str(startTime) + ".pklz", "wb")
    pickle.dump({"res": r, "algos": list(listParametersNames.keys()),
                 "datasets": datasetsDone}, f)
    f.close()
