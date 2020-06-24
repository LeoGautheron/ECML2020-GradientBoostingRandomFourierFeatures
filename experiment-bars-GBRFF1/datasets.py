#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
from collections import OrderedDict
import time
from sklearn.preprocessing import OneHotEncoder

import numpy as np

f = "../datasets/"


def loadCsv(path):
    data = []
    with open(path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(np.array(row))
    data = np.array(data)
    (n, d) = data.shape
    return data, n, d


def oneHotEncodeColumns(data, columnsCategories):
    dataCategories = data[:, columnsCategories]
    dataEncoded = OneHotEncoder(sparse=False).fit_transform(dataCategories)
    columnsNumerical = []
    for i in range(data.shape[1]):
        if i not in columnsCategories:
            columnsNumerical.append(i)
    dataNumerical = data[:, columnsNumerical]
    return np.hstack((dataNumerical, dataEncoded)).astype(float)


def loadAustralian():
    data, n, d = loadCsv(f + 'australian/australian.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadBankmarketing():
    data, n, d = loadCsv(f + 'bankmarketing/bankmarketing.csv')
    rawX = data[:, np.arange(0, d-1)]
    rawX = oneHotEncodeColumns(rawX, [1, 2, 3, 4, 6, 7, 8, 10, 15])
    rawY = data[:, d-1]
    rawY[rawY == "no"] = "-1"
    rawY[rawY == "yes"] = "1"
    rawY = rawY.astype(int)
    return rawX, rawY


def loadBalance():
    data, n, d = loadCsv(f + 'balance/balance.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0]
    rawY = rawY.astype(np.dtype(('U10', 1)))
    rawY[rawY != 'L'] = "-1"
    rawY[rawY == 'L'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadBupa():
    data, n, d = loadCsv(f + 'bupa/bupa.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadGerman():
    data, n, d = loadCsv(f + 'german/german.data')
    rawX = data[:, np.arange(1, d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadHeart():
    data, n, d = loadCsv(f + 'heart/heart.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY = rawY.astype(int)
    rawY[rawY != 2] = -1
    rawY[rawY == 2] = 1
    return rawX, rawY


def loadIonosphere():
    data, n, d = loadCsv(f + 'ionosphere/ionosphere.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'b'] = '-1'
    rawY[rawY == 'b'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadNewthyroid():
    data, n, d = loadCsv(f + 'newthyroid/newthyroid.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY < 2] = -1
    rawY[rawY >= 2] = 1
    return rawX, rawY


def loadOccupancy():
    data, n, d = loadCsv(f + 'occupancy/occupancy.csv')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY == 0] = -1
    return rawX, rawY


def loadPima():
    data, n, d = loadCsv(f + 'pima/pima-indians-diabetes.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != '1'] = '-1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadSonar():
    data, n, d = loadCsv(f + 'sonar/sonar.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'R'] = '-1'
    rawY[rawY == 'R'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadSpambase():
    data, n, d = loadCsv(f + 'spambase/spambase.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


def loadSplice():
    data, n, d = loadCsv(f + 'splice/splice.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0].astype(int)
    rawY[rawY == 1] = 2
    rawY[rawY == -1] = 1
    rawY[rawY == 2] = -1
    return rawX, rawY


def loadVehicle():
    data, n, d = loadCsv(f + 'vehicle/vehicle.data')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != "van"] = '-1'
    rawY[rawY == "van"] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadWdbc():
    data, n, d = loadCsv(f + 'wdbc/wdbc.dat')
    rawX = data[:, np.arange(d-1)].astype(float)
    rawY = data[:, d-1]
    rawY[rawY != 'M'] = '-1'
    rawY[rawY == 'M'] = '1'
    rawY = rawY.astype(int)
    return rawX, rawY


def loadWine():
    data, n, d = loadCsv(f + 'wine/wine.data')
    rawX = data[:, np.arange(1, d)].astype(float)
    rawY = data[:, 0].astype(int)
    rawY[rawY != 1] = -1
    return rawX, rawY


d = OrderedDict()
s = time.time()

d["wine"] = loadWine()                     #    178  33.15%
d["sonar"] = loadSonar()                   #    208  46.64%
d["newthyroid"] = loadNewthyroid()         #    215  30.23%
d["heart"] = loadHeart()                   #    270  44.44%
d["bupa"] = loadBupa()                     #    345  42.03%
d["iono"] = loadIonosphere()               #    351  35.90%
d["wdbc"] = loadWdbc()                     #    569  37.26%
d["balance"] = loadBalance()               #    625  46.08%
d["australian"] = loadAustralian()         #    690  44.49%
d["pima"] = loadPima()                     #    768  34.90%
d["vehicle"] = loadVehicle()               #    846  23.52%
d["german"] = loadGerman()                 #   1000  30.00%
d["splice"] = loadSplice()                 #   3175  46.64%
d["spambase"] = loadSpambase()             #   4597  39.42%
d["occupancy"] = loadOccupancy()           #  20560  23.10%
d["bankmarketing"] = loadBankmarketing()   #  45211  11.70%

print("Data loaded in {:5.2f}".format(time.time()-s))
