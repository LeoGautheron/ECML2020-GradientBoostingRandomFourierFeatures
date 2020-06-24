#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from sklearn.datasets import make_moons

from gbrff import gbrff

# seed = int(sys.argv[1])
seed = 55
np.random.seed(seed)
random.seed(seed)

n_samples = 100
Sx, Sy = make_moons(n_samples, shuffle=False, noise=0.05)
Sy[Sy == 0] = -1
trans = -np.mean(Sx, axis=0)
Sx = 2 * (Sx + trans)
xMin = min(Sx[:, 0])-2.5
xMax = max(Sx[:, 0])+2.5
yMin = min(Sx[:, 1])-2.5
yMax = max(Sx[:, 1])+2.5

step = .05
xx, yy = np.meshgrid(np.arange(xMin, xMax+step, step),
                     np.arange(yMin, yMax+step, step))
mesh = np.c_[xx.ravel(), yy.ravel()]


fontsize = 26
matplotlib.rcParams.update({'font.size': fontsize})
algo = "GBRFF"

rc('text', usetex=True)  # use same font as Latex
matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
plt.rcParams['svg.fonttype'] = 'none'  # selectable text in SVG

size1 = 200
size2 = 1600

c1 = "#3C4EC2"
c2 = "#B50927"

iterations = [1, 5, 10]
rows = 2
columns = len(iterations)
fig = plt.figure(1, figsize=(columns*8, rows*5))
subplotNumber = 0
for nbIter in iterations:
    if algo == "GBRFF":
        clf = gbrff(init="Mean",
                    gamma=1/Sx.shape[1],
                    T=nbIter,
                    learning_rate=1,
                    randomState=np.random.RandomState(seed))
    clf.fit(Sx, Sy, mesh)
    pred = clf.predict(Sx)
    accuracy = 100 * float(sum(pred == Sy)) / len(Sy)
    subplotNumber += 1
    ax = fig.add_subplot(rows, columns, subplotNumber)
    ax.scatter(Sx[:, 0], Sx[:, 1], c=Sy, edgecolor='black', label="Training points",
               linewidth=1, marker="o", s=[size1] * len(Sx),
               cmap=ListedColormap([c1, c2]), zorder=1)

    if nbIter > 1:
        landmarks = np.array([clf.xts[t]
                              for t in range(len(clf.xts)-1)])
        ax.scatter(landmarks[:, 0], landmarks[:, 1], edgecolor='black',
                   label="Previous landmarks",
                   c="#707070", linewidth=2, marker="P",
                   s=[size2] * len(landmarks), zorder=1)
    lastLandmarks = np.array([clf.xts[t]
                              for t in range(len(clf.xts)-1, len(clf.xts))])
    ax.scatter(lastLandmarks[:, 0], lastLandmarks[:, 1], edgecolor='black',
               c="white", label="Last landmark", linewidth=2, marker="P",
               s=[size2] * len(lastLandmarks), zorder=1)

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_xticks([])
    ax.set_yticks([])

    if nbIter == iterations[1]:
        ax.set_title(("Loss " + r"$f_{\boldsymbol{\omega}}$" +
                      " depending on landmark positions" +
                      "\nIteration " + str(nbIter)))
    else:
        ax.set_title("Iteration " + str(nbIter))
    Z = np.array(clf.debug[-1]).reshape(xx.shape)
    minZ = np.min(Z)
    maxZ = np.max(Z)
    stepZ = (maxZ-minZ) / 1000
    levels = np.arange(minZ, maxZ+stepZ, stepZ)

    ct = ax.contourf(xx, yy, Z, levels, cmap=plt.cm.summer.reversed(), zorder=0)
    fig.colorbar(ct, ax=ax, orientation='vertical', format='%.1f', pad=0.01,
                 ticks=[np.min(Z), (np.min(Z)+np.max(Z))/2, np.max(Z)])
    if nbIter == iterations[-1]:
        leg2 = ax.legend(framealpha=0.3,
                         ncol=3, handletextpad=0.7, handlelength=0.8,
                         bbox_to_anchor=(0.3, -1.35))
        leg2.legendHandles[0].set_color([c1])

    pred = clf.predict(Sx)
    accuracy = 100 * float(sum(pred == Sy)) / len(Sy)

    ax = fig.add_subplot(rows, columns, subplotNumber+len(iterations))
    ax.scatter(Sx[:, 0], Sx[:, 1], c=Sy, edgecolor='black', label="Training points",
               linewidth=1, marker="o", s=[size1] * len(Sx),
               cmap=ListedColormap([c1, c2]), zorder=2)

    if nbIter > 1:
        landmarks = np.array([clf.xts[t]
                              for t in range(len(clf.xts)-1)])
        ax.scatter(landmarks[:, 0], landmarks[:, 1], edgecolor='black',
                   c="#707070", linewidth=2, marker="P",
                   s=[size2] * len(landmarks), zorder=3)
    lastLandmarks = np.array([clf.xts[t]
                              for t in range(len(clf.xts)-1, len(clf.xts))])
    ax.scatter(lastLandmarks[:, 0], lastLandmarks[:, 1], edgecolor='black',
               c="white", label="Last landmark", linewidth=2, marker="P",
               s=[size2] * len(lastLandmarks), zorder=3)

    ax.set_xlim(xMin, xMax)
    ax.set_ylim(yMin, yMax)
    ax.set_xticks([])
    ax.set_yticks([])
    title = ""
    if nbIter == iterations[1]:
        title += "Overall decision boundary.\n"
    if accuracy > 99.9:
        title += "Accuracy: {:3.0f}\%".format(accuracy)
    else:
        title += "Accuracy: {:2.1f}\%".format(accuracy)
    ax.set_title(title)
    Z = clf.decision_function(mesh).reshape(xx.shape)
    minZ = np.min(Z)
    maxZ = np.max(Z)
    stepZ = (maxZ-minZ) / 1000
    levels = np.arange(minZ, maxZ+stepZ, stepZ)
    ct = ax.contourf(xx, yy, Z, levels, cmap=plt.cm.coolwarm, zorder=0)
    ax.contour(xx, yy, Z, levels, cmap=plt.cm.coolwarm, zorder=0)
    ax.contour(xx, yy, Z, [0], colors="white", zorder=1, linewidths=4)
    fig.colorbar(ct, ax=ax, orientation='vertical', format='%.1f', pad=0.01,
                 ticks=[np.min(Z), (np.min(Z)+np.max(Z))/2, np.max(Z)])

fig.subplots_adjust(wspace=-0.02, hspace=0.35)
fig.savefig("moons" + algo + "lossPlacingLandmark" + str(seed) + ".png",
            bbox_inches="tight")
# Free ram usage with these 4 last lines
for ax in fig.axes:
    ax.cla()
fig.clf()
plt.close(fig)
