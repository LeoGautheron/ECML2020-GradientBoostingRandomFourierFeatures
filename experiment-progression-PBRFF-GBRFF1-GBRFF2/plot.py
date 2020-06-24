#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import gzip
import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


ticks = np.arange(1, 51, 1)
rReal = []
datasets = []
algos = []
for filename in glob.glob("results1/resReal*.pklz"):
    f = gzip.open(filename, "rb")
    res = pickle.load(f)
    f.close()
    rReal.append(res["res"])
    if datasets == [] or len(res["datasets"]) < len(datasets):
        datasets = res["datasets"]
    algos = res["algos"]


def getTestResultsValidation(r, algos):
    res = []
    for i in range(len(r)):
        res.append({})
        for da in datasets:
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
                res[i][da]["test"][a] = r[i][da]["test"][a][bestP][0]
                res[i][da]["time"][a] = r[i][da]["test"][a][bestP][1]
    return res


def getMean(rReal, algos):
    mrReal = {}
    for da in datasets:
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
                mrReal[da]["test"][a][s] = ((mean*100, std*100))
            mrReal[da]["time"][a] = np.sum([rReal[i][da]["time"][a]
                                            for i in range(len(rReal))])

    all = "Mean"
    mrReal[all] = {"test": {}, "time": {}}
    mrReal[all]["test"] = {a:
                           {s:
                            (np.mean([mrReal[da]["test"][a][s][0]
                                      for da in datasets]),
                             np.mean([mrReal[da]["test"][a][s][1]
                                      for da in datasets]))
                            for s in ["train", "test"]}
                           for a in algos}
    mrReal[all]["time"] = {a: np.sum([mrReal[da]["time"][a]
                                      for da in datasets]) for a in algos}
    return mrReal


mr = getMean(getTestResultsValidation(rReal, algos), algos)
del rReal

colors = {"PBRFF K=10": '#1f77b4',
          "GBRFF0.5 K=10": '#ff6ea0',
          "GBRFF1 K=20": '#ffa512',
          "GBRFF1 K=10": '#ff7f0e',
          "GBRFF1 K=5": '#b2590a',
          "GBRFF1 K=1": '#663306',
          "GBRFF1.5": '#958f1d',
          "GBRFF2": '#2ca02c'}

linestyles = {"PBRFF K=10": (0, (1, 1)),
              "GBRFF0.5 K=10": (0, (3, 1, 1, 1)),
              "GBRFF1 K=20": (0, (3, 1, 1, 1)),
              "GBRFF1 K=10": (0, (3, 1, 1, 1)),
              "GBRFF1 K=5": (0, (3, 1, 1, 1)),
              "GBRFF1 K=1": (0, (3, 1, 1, 1)),
              "GBRFF1.5":  (0, (1, 1)),
              "GBRFF2": "solid"}

for algosNames in [["PBRFF K=10", "GBRFF0.5 K=10", "GBRFF1 K=10"],
                   ["GBRFF1 K=20", "GBRFF1 K=10", "GBRFF1 K=5", "GBRFF1 K=1"],
                   ["GBRFF1 K=1", "GBRFF1 K=20", "GBRFF1.5", "GBRFF2"],
                   ["PBRFF K=10", "GBRFF1 K=10", "GBRFF2"]]:
    accuracies = {a: [] for a in algosNames}
    stds = {a: [] for a in algosNames}
    computationTimes = {a: [] for a in algosNames}

    for i in ticks:
        for a in algosNames:
            if a == "PBRFF K=10":
                realAlgoName = "PBRFF;" + str(i) + ";10"
            elif a == "GBRFF0.5 K=10":
                realAlgoName = "GBRFF05;" + str(i) + ";10"
            elif a == "GBRFF1 K=20":
                realAlgoName = "GBRFF1;" + str(i) + ";20"
            elif a == "GBRFF1 K=10":
                realAlgoName = "GBRFF1;" + str(i) + ";10"
            elif a == "GBRFF1 K=5":
                realAlgoName = "GBRFF1;" + str(i) + ";5"
            elif a == "GBRFF1 K=1":
                realAlgoName = "GBRFF1;" + str(i) + ";1"
            elif a == "GBRFF1.5":
                realAlgoName = "GBRFF15;" + str(i)
            elif a == "GBRFF2":
                realAlgoName = "GBRFF2;" + str(i)
            accuracies[a].append(mr["Mean"]["test"][realAlgoName]["test"][0])
            stds[a].append(mr["Mean"]["test"][realAlgoName]["test"][1])
            computationTimes[a].append(mr["Mean"]["time"][realAlgoName])
    fontsize = 20
    matplotlib.rcParams.update({'font.size': fontsize})
    fig = plt.figure(1, figsize=(16, 5))
    ax = fig.add_subplot(1, 2, 1)
    minAcc = 100
    maxAcc = 0
    for a in algosNames:
        resMin = [accuracies[a][i] - stds[a][i]
                  for i in range(len(accuracies[a]))]
        resMax = [accuracies[a][i] + stds[a][i]
                  for i in range(len(accuracies[a]))]
        minAcc = min(minAcc, min(resMin))
        maxAcc = max(maxAcc, max(resMax))
        ax.plot(ticks, accuracies[a], label=a, lw=4, color=colors[a],
                linestyle=linestyles[a])
        ax.fill_between(ticks, resMin, resMax, alpha=0.3, color=colors[a])
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlim([0.5, ticks[-1]])
    ax.set_xticklabels([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel("Number of landmarks")
    ax.set_yticks([60, 65, 70, 75, 80, 85, 90])
    ax.set_ylim([minAcc, maxAcc])
    ax.set_ylabel("Accuracy")
    offset = 1.5
    if len(algosNames) == 3:
        offset = 1.95
    if len(algosNames) == 4:
        offset = 2.1
    leg = ax.legend(framealpha=1, ncol=4, handlelength=1.0,
                    bbox_to_anchor=(offset, -0.15))
    for line in leg.get_lines():
        line.set_linewidth(2)
    ax.grid(True)
    # Plot computation time
    ax = fig.add_subplot(1, 2, 2)
    maxTime = 0
    for a in algosNames:
        ax.plot(ticks, computationTimes[a], label=a, lw=4, color=colors[a],
                linestyle=linestyles[a])
        maxTime = max(maxTime, max(computationTimes[a]))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlim([0.5, ticks[-1]])
    ax.set_xticklabels([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel("Number of landmarks")
    ax.set_ylabel("Computation time in seconds")
    ax.set_ylim([0, maxTime])
    ax.grid(True)
    fig.savefig("graph"+str.join(".", algosNames).replace(" ", "")+".pdf",
                bbox_inches="tight")
    # Free ram usage with these 4 last lines
    for ax in fig.axes:
        ax.cla()
    fig.clf()
    plt.close(fig)

    # Plot accuracy divided by computation time
    fig = plt.figure(1, figsize=(8, 5))
    ax = fig.add_subplot(1, 1, 1)
    maxRat = 0
    for a in algosNames:
        accOverTime = np.array(accuracies[a]) / np.array(computationTimes[a])
        ax.plot(ticks, accOverTime, label=a, lw=4,
                color=colors[a], linestyle=linestyles[a])
        maxRat = max(maxRat, max(accOverTime))
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlim([0.5, ticks[-1]])
    ax.set_xticklabels([1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.set_xlabel("Number of landmarks")
    ax.set_ylabel("Accuracy divided by time")
    ax.set_ylim([0, maxRat])
    ax.grid(True)
    offset = 0.8
    if len(algosNames) == 3:
        offset = 0.75
    if len(algosNames) == 4:
        offset = 0.95
    leg = ax.legend(framealpha=1, ncol=2, handlelength=1.0,
                    bbox_to_anchor=(offset, -0.15))
    for line in leg.get_lines():
        line.set_linewidth(2)
    fig.savefig("graphRatio"+str.join(".", algosNames).replace(" ", "")+".pdf",
                bbox_inches="tight")
    # Free ram usage with these 4 last lines
    for ax in fig.axes:
        ax.cla()
    fig.clf()
    plt.close(fig)
