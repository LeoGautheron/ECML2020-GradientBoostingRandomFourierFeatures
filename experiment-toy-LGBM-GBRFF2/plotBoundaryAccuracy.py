#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import glob
import gzip
import pickle

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

rAcc = []
n_samples = []
for filename in glob.glob("res*.pklz"):
    f = gzip.open(filename, "rb")
    res = pickle.load(f)
    f.close()
    rAcc.append(res["res"])
    n_samples2 = res["n_samples"]
    if len(n_samples) == 0 or len(n_samples2) < len(n_samples):
        n_samples = n_samples2
    algos = res["algos"]
    datasets = res["datasets"]
r = rAcc[0]  # 14
fontsize = 12
matplotlib.rcParams.update({'font.size': fontsize})
c1 = "#3C4EC2"
c2 = "#B50927"
size1 = 100
rows = len(algos)+1
columns = len(datasets)
fig = plt.figure(1, figsize=(columns*4, rows*4))
minY = 100
maxY = 0
for dnb, dataset in enumerate(datasets):
    if dataset == "swiss":
        nPlot = n_samples[len(n_samples)//2]  # 400
    elif dataset == "circles":
        nPlot = n_samples[len(n_samples)//2]  # 400
    elif dataset == "board":
        nPlot = n_samples[len(n_samples)//2]  # 250
    for anb, a in enumerate(algos):
        xMin, xMax = r[nPlot][dataset]["xMin"], r[nPlot][dataset]["xMax"]
        yMin, yMax = r[nPlot][dataset]["yMin"], r[nPlot][dataset]["yMax"]
        xx, yy = r[nPlot][dataset]["xx"], r[nPlot][dataset]["yy"]
        mesh = r[nPlot][dataset]["mesh"]
        Xtrain, Xtest = r[nPlot][dataset]["Xtrain"], r[nPlot][dataset]["Xtest"]
        ytrain, ytest = r[nPlot][dataset]["ytrain"], r[nPlot][dataset]["ytest"]
        ax = fig.add_subplot(rows, columns, anb*len(datasets)+dnb+1)
        ax.scatter(Xtrain[:, 0], Xtrain[:, 1], c=ytrain, edgecolor='black',
                   label="Training points", linewidth=0.5, marker="o",
                   s=[size1]*len(Xtrain),
                   cmap=ListedColormap([c1, c2]), zorder=1)
        ax.scatter(Xtest[:, 0], Xtest[:, 1], c=ytest, edgecolor='black',
                   label="Testing points", linewidth=0.5, marker="P",
                   s=[size1]*len(Xtest),
                   cmap=ListedColormap([c1, c2]), zorder=1)
        ax.set_xlim(xMin, xMax)
        ax.set_ylim(yMin, yMax)
        if dataset == datasets[0]:
            ax.set_ylabel(a.split(";")[0], fontsize=fontsize)
        ax.set_xticks([])
        ax.set_yticks([])
        title = ""
        if a == algos[0] and dataset == datasets[1]:
            title += "Overall decision boundary\n"
        if a == algos[0]:
            title += dataset + " " + str(nPlot) + " samples"
        trainAccuracy = r[nPlot][dataset][a]["trainAccuracy"]
        testAccuracy = r[nPlot][dataset][a]["testAccuracy"]
        accuraciesLabel = ""
        if trainAccuracy > 99.9:
            accuraciesLabel += "Train {:3.0f}%".format(trainAccuracy)
        else:
            accuraciesLabel += "Train {:2.1f}%".format(trainAccuracy)
        if testAccuracy > 99.9:
            accuraciesLabel += " Test {:3.0f}%".format(testAccuracy)
        else:
            accuraciesLabel += " Test {:2.1f}%".format(testAccuracy)
        if a == "GBRFF2":
            title += accuraciesLabel
        else:
            ax.set_xlabel(accuraciesLabel, fontsize=fontsize+4)
        ax.set_title(title, fontsize=fontsize+4)
        Z = r[nPlot][dataset][a]["Z"]
        minZ = np.min(Z)
        maxZ = np.max(Z)
        if -minZ < maxZ:
            minZ = -maxZ
        elif maxZ < -minZ:
            maxZ = -minZ
        stepZ = (maxZ-minZ) / 2000
        levels = np.arange(minZ, maxZ+stepZ, stepZ)
        ct = ax.contourf(xx, yy, Z, levels, cmap=plt.cm.coolwarm, zorder=0,
                         vmin=minZ, vmax=maxZ)
        ax.contour(xx, yy, Z, levels, cmap=plt.cm.coolwarm, zorder=0)
        ax.contour(xx, yy, Z, [0], colors="white", zorder=1, linewidths=2)
        if dataset == datasets[-1] and a == algos[-1]:
            leg2 = ax.legend(framealpha=1, bbox_to_anchor=(-0.51, -1.40),
                             ncol=3, handletextpad=0.7, handlelength=0.8)
            leg2.legendHandles[0].set_color([c1])
        for n in n_samples:
            test = [rAcc[i][n][dataset][a]["testAccuracy"]
                    for i in range(len(rAcc))]
            low = np.mean(test) - np.std(test)
            high = np.mean(test) + np.std(test)
            if low < minY:
                minY = low
            if high > maxY:
                maxY = high
for dnb, dataset in enumerate(datasets):
    ax = fig.add_subplot(rows, columns, len(algos)*len(datasets)+dnb+1)
    algos = ["GBRFF2", "LGBM"]
    colors = ['#2ca02c', '#8c564b']
    linestyles = ["solid", (0, (5, 1))]
    xAxis = np.arange(len(n_samples))
    for a, color, linestyle in zip(algos, colors, linestyles):
        testAccuracies = []
        resMin = []
        resMax = []
        for n in n_samples:
            test = [rAcc[i][n][dataset][a]["testAccuracy"]
                    for i in range(len(rAcc))]
            testAccuracies.append(np.mean(test))
            resMin.append(np.mean(test)-np.std(test))
            resMax.append(np.mean(test)+np.std(test))
        ax.plot(xAxis, testAccuracies, label=a, linestyle=linestyle,
                color=color, lw=3)
        ax.fill_between(xAxis, resMin, resMax, alpha=0.3, color=color)
        if len(xAxis) > 5:
            last = len(xAxis)-1
            ticks = [0, (2*last)//5, (3*last)//5, (4*last)//5, last]
            ax.set_xticks(ticks)
            ax.set_xticklabels(n_samples[ticks])
        else:
            ax.set_xticks(xAxis)
            ax.set_xticklabels(n_samples)
        ax.set_xlim([-0.10*xAxis[-1],
                     xAxis[-1]+0.10*xAxis[-1]])
        ax.set_ylim([minY, maxY])
    ax.grid(True)
    ax.set_xlabel("Number of samples")
    if dataset == datasets[0]:
        ax.set_ylabel("Test accuracy")
    else:
        ax.set_yticklabels([])
    title = ""
    if dataset == datasets[1]:
        title += "Comparison when increasing the number of samples\n"
    title += dataset
    ax.set_title(title, fontsize=fontsize+4)
    if dataset == datasets[-1]:
        ax.legend(framealpha=1, ncol=2, bbox_to_anchor=(0.41, -0.15))
fig.subplots_adjust(wspace=0, hspace=0.25)
fig.savefig("toysLGBMGBRFF2.png", bbox_inches="tight")
# Free ram usage with these 4 last lines
for ax in fig.axes:
    ax.cla()
fig.clf()
plt.close(fig)
