#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import random
import time
import os
import sys
import gzip

import numpy as np
from sklearn.datasets import make_classification

from lightgbm import LGBMClassifier
from bmkr import BMKR
from gfc import GreedyFeatureConstruction
from pbrff import pbrff
from gbrff1 import gbrff as gbrff1
from gbrff2 import gbrff as gbrff2


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


def applyAlgo(algo, X, y):
    nbIter = 100
    K = 10
    if algo.startswith("LGBM"):
        clf = LGBMClassifier(n_estimators=nbIter,
                             max_depth=5,
                             num_leaves=2**5,
                             n_jobs=1,
                             random_state=1,
                             verbose=-1)
    if algo.startswith("BMKR"):
        clf = BMKR(T=nbIter, C=1)
    if algo.startswith("GFC"):
        clf = GreedyFeatureConstruction(numFeatures=nbIter, C=1,
                                        randomState=np.random.RandomState(1))
    if algo.startswith("PBRFF"):
        clf = pbrff(beta=1, K=K, gamma=1/X.shape[1], n_landmarks=nbIter,
                    C=1, randomState=np.random.RandomState(1))
    if algo.startswith("GBRFF1"):
        clf = gbrff1(gamma=1/X.shape[1], T=nbIter, beta=1, K=K,
                     randomState=np.random.RandomState(1))
    if algo.startswith("GBRFF2"):
        clf = gbrff2(Lambda=0, gamma=1/X.shape[1],
                     T=nbIter, randomState=np.random.RandomState(1))
    t1 = time.time()
    clf.fit(X, y)
    t2 = time.time()
    clf.predict(X)
    t3 = time.time()
    return (t2-t1, t3-t2)


algos = ["LGBM", "BMKR", "GFC", "PBRFF", "GBRFF1", "GBRFF2"]
r = {}  # All the results are stored in this dictionnary
stoppedAlgo = []
datasetsDone = []
startTime = time.time()
timeLimit = 2200  # seconds
n_samples = 100
increaseSampleMult = 1.5
while len(stoppedAlgo) != len(algos):
    n_samples = int(n_samples * increaseSampleMult)
    print(n_samples)
    datasetsDone.append(n_samples)
    X, y = make_classification(n_samples=n_samples,
                               n_classes=2,
                               random_state=1)
    if len(sys.argv) == 2:
        np.random.seed(int(sys.argv[1]))
        random.seed(int(sys.argv[1]))
    r[n_samples] = {"timeFit": {}, "timePred": {}}
    for a in algos:
        if a in stoppedAlgo:
            continue
        timeFit, timePred = applyAlgo(a, X, y)
        r[n_samples]["timeFit"][a] = timeFit
        r[n_samples]["timePred"][a] = timePred
        print(n_samples, a,
              "timeFit: {:8.2f} sec".format(timeFit),
              "timePred: {:8.2f} sec".format(timePred))
        if timeFit + timePred > timeLimit:
            stoppedAlgo.append(a)
    del X, y
    # Save the results at the end of each dataset
    if len(sys.argv) == 2:
        f = gzip.open("./results/res" + sys.argv[1] + ".pklz", "wb")
    else:
        f = gzip.open("./results/res" + str(startTime) + ".pklz", "wb")
    pickle.dump({"res": r, "algos": list(algos),
                 "datasets": datasetsDone}, f)
    f.close()
