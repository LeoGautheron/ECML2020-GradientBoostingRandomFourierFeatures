#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import gzip
import os

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc


rReal = []
datasetsReal = []
for filename in glob.glob("./results/resReal*.pklz"):
    f = gzip.open(filename, "rb")
    res = pickle.load(f)
    f.close()
    rReal.append(res["res"])

    if datasetsReal == [] or len(res["datasets"]) < len(datasetsReal):
        datasetsReal = res["datasets"]
    algos = res["algos"]


def getTestResultsValidation(r):
    res = []
    for i in range(len(r)):
        res.append({})
        for da in datasetsReal:
            res[i][da] = {"test": {}, "time": {}}
            for a in algos:
                sor = []
                for nameP in r[i][da]["valid"][a].keys():
                    perfP = []
                    for perf in r[i][da]["valid"][a][nameP]:
                        tn, fp, fn, tp = perf[-1]["test"]
                        acc = (tp+tn)/(tp+tn+fp+fn)
                        perfP.append(acc)
                    sor.append((np.mean(perfP), nameP))
                sor = sorted(sor)
                bestP = sor[-1][1]
                res[i][da]["test"][a] = r[i][da]["test"][a][bestP]
                res[i][da]["time"][a] = r[i][da]["time"][a]
    return res


def getMean(rReal):
    mrReal = {}
    for da in datasetsReal:
        mrReal[da] = {"test": {}, "time": {}}
        for a in algos:
            mrReal[da]["test"][a] = {}
            for s in ["train", "test"]:
                accuracies = []
                for i in range(len(rReal)):
                    tn, fp, fn, tp = rReal[i][da]["test"][a][-1][s]
                    acc = (tp+tn)/(tp+tn+fp+fn)
                    accuracies.append(acc)
                mean = np.mean(accuracies)
                std = np.std(accuracies)
                mrReal[da]["test"][a][s] = ((mean, std))
            mrReal[da]["time"][a] = sum([rReal[i][da]["time"][a]
                                         for i in range(len(rReal))])

    all = "Mean"
    mrReal[all] = {"test": {}, "time": {}}
    mrReal[all]["test"] = {a:
                       {s:
                        (np.mean([mrReal[da]["test"][a][s][0] for da in datasetsReal]),
                         np.mean([mrReal[da]["test"][a][s][1] for da in datasetsReal]))
                        for s in ["train", "test"]}
                       for a in algos}
    mrReal[all]["time"] = {a: sum([mrReal[da]["time"][a]
                                   for da in datasetsReal]) for a in algos}
    return mrReal


mr = getMean(getTestResultsValidation(rReal))

matplotlib.rcParams.update({'font.size': 16})
rc('text', usetex=True)  # use same font as Latex
matplotlib.rcParams['text.latex.preamble'] = [
        r'\usepackage{amsmath}'
    ]
plt.rcParams['svg.fonttype'] = 'none'  # selectable text in SVG

newAlgos = ["GBRFF1 K=20", "GBRFF1 K=10", "GBRFF1 K=5", "GBRFF1 K=1"]
newMr = {a: ([], []) for a in newAlgos}


for a in ["GBRFF1;" + str(i) + ";20" for i in [1, 5, 10, 15, 25, 50]]:
    newMr["GBRFF1 K=20"][0].append(mr["Mean"]["test"][a]["test"][0]*100)
    newMr["GBRFF1 K=20"][1].append(mr["Mean"]["test"][a]["test"][1]*100)

for a in ["GBRFF1;" + str(i) + ";10" for i in [2, 10, 20, 30, 50, 100]]:
    newMr["GBRFF1 K=10"][0].append(mr["Mean"]["test"][a]["test"][0]*100)
    newMr["GBRFF1 K=10"][1].append(mr["Mean"]["test"][a]["test"][1]*100)

for a in ["GBRFF1;" + str(i) + ";5" for i in [4, 20, 40, 60, 100, 200]]:
    newMr["GBRFF1 K=5"][0].append(mr["Mean"]["test"][a]["test"][0]*100)
    newMr["GBRFF1 K=5"][1].append(mr["Mean"]["test"][a]["test"][1]*100)

for a in ["GBRFF1;" + str(i) + ";1" for i in [20, 100, 200, 300, 500, 1000]]:
    newMr["GBRFF1 K=1"][0].append(mr["Mean"]["test"][a]["test"][0]*100)
    newMr["GBRFF1 K=1"][1].append(mr["Mean"]["test"][a]["test"][1]*100)

labels = [20, 100, 200, 300, 500, 1000]

# define the colors used
ini = (0xff, 0x7f, 0x0e)
sec = []
for v in ini:
    sec.append(int(v))
# 3 higher, 9 lower
colors = []
fcts = [0.15, 0.07, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15]
for i, fct in zip([2, 1, 0, -1, -2, -3, -4, -5], fcts):
    color = "#"
    for v in sec:
        nv = round(i*fct*v + v)
        if nv > 255:
            nv = 255
        if nv < 0:
            nv = 0
        st = str(hex(nv))[2:]
        if len(st) == 1:
            st = "0" + st
        color += st
    colors.append(color)

rc('text', usetex=True)  # use same font as Latex
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rcParams['svg.fonttype'] = 'none'  # selectable text in SVG

patterns = ('-', '+', 'x', r'\\')
index = np.arange(len(labels))
fig = plt.figure(1, figsize=(13.5, 3))
ax = fig.add_subplot(1, 1, 1)
bar_width = 0.95 / len(newAlgos)
matplotlib.rcParams['font.size'] = 16
opacity = 1.0
for j, a in enumerate(newAlgos):
    c1 = colors[2*j]
    c2 = colors[2*j+1]
    print(a, c1)
    ax.bar(index+j*bar_width, newMr[a][0], bar_width, yerr=newMr[a][1],
           color=c1, alpha=opacity, label=a, zorder=0,
           error_kw=dict(ecolor=c2, lw=2, capsize=5, capthick=2))
    for i, v in enumerate(newMr[a][0]):
        vText = "{:5.2f}".format(v)
        if v >= 99.9999:
            vText = "100.0"
        ax.text(i-0.11+j*bar_width, v-4.2, vText, color='black',
                size=12, zorder=1)
ax.legend(loc="upper left", ncol=len(newAlgos))
ax.grid(False)
ax.set_ylim(70, 97)
ax.set_yticks([70, 75, 80, 85, 90])
ax.set_ylabel("Accuracy")
ax.set_xlabel(r"Total number of random features used in the whole process ($T\times K$)")
ax.set_xlim([index[0]-0.2, index[-1]+0.9])
ax.set_xticks(index+(len(newAlgos)-1)*bar_width/2)  # +len(algos)*bar_width
ax.set_xticklabels(labels)  # , rotation=45)

fig.subplots_adjust(wspace=0.02, hspace=0.35)
plt.savefig("bars_GBRFF1.pdf", bbox_inches="tight")
plt.close(fig)
