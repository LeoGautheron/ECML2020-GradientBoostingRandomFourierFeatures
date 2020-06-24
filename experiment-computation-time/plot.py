#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import gzip

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import rc


matplotlib.rcParams.update({'font.size': 16})
r = []
datasets = []
for filename in glob.glob("./results/res*.pklz"):
    f = gzip.open(filename, "rb")
    res = pickle.load(f)
    f.close()
    r.append(res["res"])

    if datasets == [] or len(res["datasets"]) < len(datasets):
        datasets = res["datasets"]
    algos = res["algos"]


mr = {}
for a in algos:
    mr[a] = {"timeFit": [], "timePred": []}
    for da in datasets:
        timeFitAlgo = []
        timePredAlgo = []
        for i in range(len(r)):
            if a in r[i][da]["timeFit"]:
                timeFitAlgo.append(r[i][da]["timeFit"][a])
                timePredAlgo.append(r[i][da]["timePred"][a])
        if len(timeFitAlgo) == len(r):
            mr[a]["timeFit"].append(np.mean(timeFitAlgo))
            mr[a]["timePred"].append(np.mean(timePredAlgo))
algos = ["BMKR", "GFC", "PBRFF", "GBRFF1", "GBRFF2", "LGBM"]
colors = ['#d62728', '#9467bd', '#1f77b4', '#ff7f0e', '#2ca02c', '#8c564b']
linestyles = [(0, (1, 2)), (0, (3, 1, 1, 1, 1, 1)), (0, (1, 1)), (0, (3, 1, 1, 1)), "solid", (0, (5, 1))]
fig = plt.figure(1, figsize=(13.5, 3))
ax = fig.add_subplot(1, 1, 1)
startIdx = 6
datasets = datasets[startIdx:]
#PBRFF '#1f77b4' (0, (1, 1))
#GBRFF1 '#ff7f0e' (0, (3, 1, 1, 1))
#GBRFF2 '#2ca02c' "solid"
#BMKR '#d62728' (0, (1, 2))
#GFC '#9467bd' (0, (3, 1, 1, 1, 1, 1))
#LGBM '#8c564b' (0, (5, 1))
for a, color, linestyle in zip(algos, colors, linestyles):
    if a == "LGBM":
        timeFit = np.array(mr[a]["timeFit"])[startIdx:-1]
        timePred = np.array(mr[a]["timePred"])[startIdx:-1]
    else:
        timeFit = np.array(mr[a]["timeFit"])[startIdx:]
        timePred = np.array(mr[a]["timePred"])[startIdx:]
    xAxis = np.log([datasets[i] for i in range(len(timeFit))])
    ax.plot(xAxis, timeFit+timePred, label=a, lw=6, color=color,
            linestyle=linestyle)
    ax.set_xlim([xAxis[0], xAxis[-1]])
    ax.set_xticks([xAxis[i]
                   for i in range(0, len(xAxis), 2)])
    ax.set_xticklabels([format(datasets[i], ",")
                        for i in range(0, len(timeFit), 2)])
    plt.xticks(rotation=15)
leg = ax.legend(loc="upper left", ncol=7, bbox_to_anchor=(0.005, 1.25),
                scatterpoints=2)
for line in leg.get_lines():
    line.set_linewidth(3.2)
ax.set_xlabel("Number of samples")
ax.set_ylabel("Time in seconds")
ax.set_ylim([0, 1000])
ax.grid()
plt.savefig("computationTime.pdf", bbox_inches="tight")
plt.close(fig)
