#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob
import pickle
import os
import gzip
import subprocess

import numpy as np

import sys
from subprocess import call

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
algos = ["BMKR;100", "GFC;100", "PBRFF;100;10", "GBRFF1;100;10", "LGBM;100", "GBRFF2;100"]


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


def latex(rReal):
    if not os.path.exists("latex"):
        os.makedirs("latex")

    f = open('latex/doc.tex', 'w')
    sys.stdout = f
    print(r"\documentclass[a4paper, 12pt]{article}")
    print(r"\usepackage[french]{babel}")
    print(r"\usepackage[T1]{fontenc}")
    print(r"\usepackage{amssymb} ")
    print(r"\usepackage{amsmath}")
    print(r"\usepackage[utf8]{inputenc}")
    print(r"\usepackage{graphicx}")
    print(r"\usepackage{newtxtext}")
    print(r"\usepackage{booktabs}")
    print(r"\usepackage{multirow}")

    print(r"\begin{document}")
    mrReal = getMean(getTestResultsValidation(rReal))

    print(r"\begin{table*}")
    print(r"\resizebox{1.0\textwidth}{!}{\begin{tabular}{l ", end="")
    for a in algos:
        print(" c ", end="")
    print("}")
    print(r"\toprule")

    print("{:12}".format("Dataset"), end="")
    ranks = {}
    for a in algos:
        print(r"&  ", end="")
        print("{:11} ".format(a.replace("_", "\\_").split(";")[0]), end="")
        ranks[a] = 0
    print(r"\\")

    print(r"\midrule")
    for da in datasetsReal + ["Mean"]:
        if da == "Mean":
            print(r"\midrule")
        print("{:12}".format(da), end="")

        order = list(reversed(
                   np.argsort([mrReal[da]["test"][a]["test"][0] for a in algos])))
        best = algos[order[0]]

        for i, idx in enumerate(order):
            ranks[algos[idx]] += 1+i

        best = algos[np.argmax([mrReal[da]["test"][a]["test"][0] for a in algos])]
        for a in algos:
            b1 = ""
            b2 = ""
            if a == best:
                b1 = r"\textbf{"
                b2 = "}"
            print("&  " + b1 +
                  "{:4.1f}".format(mrReal[da]["test"][a]["test"][0]*100) +
                  b2 + " $\\pm$ {:4.1f}".format(
                                        mrReal[da]["test"][a]["test"][1]*100),
                  end="")
        print(r"\\")
        if da == "Mean":
            print("Average Rank", end="")
            for a in algos:
                print(r"&      ", end="")
                print("{:1.2f} ".format(ranks[a]/(len(datasetsReal)+1)), end="")
            print(r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}}")
    print(r"\end{table*}")

    print(r"\end{document}")
    f.close()

    call(["pdflatex", "-output-directory=latex", "latex/doc.tex"])
    os.remove("latex/doc.aux")
    os.remove("latex/doc.log")
    subprocess.Popen(["okular latex/doc.pdf"], shell=True)


latex(rReal)
