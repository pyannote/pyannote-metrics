#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016 CNRS

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr


import matplotlib
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn.metrics
import numpy as np

def plot_distributions(y_true, scores, save_to, xlim=None, nbins=100, ymax=3.):
    """Scores distributions

    This function will create (and overwrite) the following files:
        - {save_to}.scores.png
        - {save_to}.scores.eps

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    save_to : str
        Files path prefix
    """

    plt.figure(figsize=(12, 12))
    bins = np.linspace(xlim[0], xlim[1], nbins)
    plt.hist(scores[y_true], bins=bins, color='g', alpha=0.5, normed=True)
    plt.hist(scores[~y_true], bins=bins, color='r', alpha=0.5, normed=True)

    # TODO heuristic to estimate ymax from nbins and xlim
    plt.ylim(0, ymax)
    plt.tight_layout()
    plt.savefig(save_to + '.scores.png', dpi=150)
    plt.savefig(save_to + '.scores.eps')
    plt.close()

    return True

def plot_det_curve(y_true, scores, save_to):
    """DET curve

    This function will create (and overwrite) the following files:
        - {save_to}.det.png
        - {save_to}.det.eps
        - {save_to}.det.txt

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    save_to : str
        Files path prefix.

    Returns
    -------
    eer : float
        Equal error rate
    """

    # compute false positive and false negative rates
    # (a.k.a. false alarm and false rejection rates)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true,
                                                     scores,
                                                     pos_label=True)
    fnr = 1 - tpr

    # estimate equal error rate
    eer_index = np.where(fpr > fnr)[0][0]
    eer = .25 * (fpr[eer_index-1] + fpr[eer_index] +
                 fnr[eer_index-1] + fnr[eer_index])

    plt.figure(figsize=(12, 12))
    plt.loglog(fpr, fnr, 'b')
    plt.loglog([eer], [eer], 'bo')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.xlim(1e-2, 1.)
    plt.ylim(1e-2, 1.)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_to + '.det.png', dpi=150)
    plt.savefig(save_to + '.det.eps')
    plt.close()

    txt = save_to + '.det.txt'
    line = '{t:.6f} {fp:.6f} {fn:.6f} {mark:s}\n'
    with open(txt, 'w') as f:
        for i, (t, fp, fn) in enumerate(zip(thresholds, fpr, fnr)):
            mark = 'eer' if i == eer_index else '---'
            f.write(line.format(t=t, fp=fp, fn=fn, mark=mark))

    return eer

def plot_precision_recall_curve(y_true, scores, save_to):

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        y_true, scores, pos_label=True)

    auc = sklearn.metrics.auc(precision, recall, reorder=True)

    plt.figure(figsize=(12, 12))
    plt.plot(recall, precision, 'b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(save_to + '.precision_recall.png', dpi=150)
    plt.savefig(save_to + '.precision_recall.eps')
    plt.close()

    return auc
