#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2016-2017 CNRS

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


import sklearn.metrics
import numpy as np


def det_curve(y_true, scores, distances=False):
    """DET curve

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    distances : boolean, optional
        When True, indicate that `scores` are actually `distances`

    Returns
    -------
    fpr : numpy array
        False alarm rate
    fnr : numpy array
        False rejection rate
    thresholds : numpy array
        Corresponding thresholds
    eer : float
        Equal error rate
    """

    if distances:
        scores = -scores

    # compute false positive and false negative rates
    # (a.k.a. false alarm and false rejection rates)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        y_true, scores, pos_label=True)
    fnr = 1 - tpr
    if distances:
        thresholds = -thresholds

    # estimate equal error rate
    eer_index = np.where(fpr > fnr)[0][0]
    eer = .25 * (fpr[eer_index-1] + fpr[eer_index] +
                 fnr[eer_index-1] + fnr[eer_index])

    return fpr, fnr, thresholds, eer


def precision_recall_curve(y_true, scores, distances=False):
    """Precision-recall curve

    Parameters
    ----------
    y_true : (n_samples, ) array-like
        Boolean reference.
    scores : (n_samples, ) array-like
        Predicted score.
    distances : boolean, optional
        When True, indicate that `scores` are actually `distances`

    Returns
    -------
    precision : numpy array
        Precision
    recall : numpy array
        Recall
    thresholds : numpy array
        Corresponding thresholds
    auc : float
        Area under curve

    """

    if distances:
        scores = -scores

    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(
        y_true, scores, pos_label=True)

    if distances:
        thresholds = -thresholds

    auc = sklearn.metrics.auc(precision, recall, reorder=True)

    return precision, recall, thresholds, auc
