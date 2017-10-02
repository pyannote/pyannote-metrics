#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2017 CNRS

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

from __future__ import unicode_literals

import numpy as np
from .base import BaseMetric
from .binary_classification import det_curve, precision_recall_curve
from pyannote.core import Segment, Annotation
from pyannote.core import SlidingWindowFeature


class LowLatencySpeakerSpotting(BaseMetric):
    """Evaluation of low-latency speaker spotting

    Parameters
    ----------
    thresholds : (n_thresholds, ) array
        Detection thresholds.
    """

    @classmethod
    def metric_name(cls):
        return "Low-latency speaker spotting 2"

    @classmethod
    def metric_components(cls):
        return {
            'target': 0.,
            'non_target': 0.,
            'true_positive': 0.,
            'false_positive': 0.,
            'true_negative': 0.,
            'false_negative': 0.}

    def __init__(self, thresholds=None):
        super(LowLatencySpeakerSpotting, self).__init__(parallel=False)
        self.thresholds = np.sort(thresholds)

    def compute_metric(self, detail):
        return None

    def compute_components(self, reference, hypothesis, **kwargs):
        """

        Parameters
        ----------
        reference : Timeline or Annotation
        hypothesis : SlidingWindowFeature or (time, score) iterable
        """

        if isinstance(hypothesis, SlidingWindowFeature):
            hypothesis = [(window.end, value) for window, value in hypothesis]
        timestamps, scores = zip(*hypothesis)

        # pre-compute latencies
        speaker_latency = np.NAN * np.ones((len(timestamps), 1))
        absolute_latency = np.NAN * np.ones((len(timestamps), 1))
        if isinstance(reference, Annotation):
            reference = reference.get_timeline(copy=False)
        if reference:
            first_time = reference[0].start
            for i, t in enumerate(timestamps):
                so_far = Segment(first_time, t)
                speaker_latency[i] = reference.crop(so_far).duration()
                absolute_latency[i] = max(0, so_far.duration)
            # TODO | speed up latency pre-computation

        # for every threshold, compute when (if ever) alarm is triggered
        maxcum = (np.maximum.accumulate(scores)).reshape((-1, 1))
        triggered = maxcum > self.thresholds
        indices = np.array([np.searchsorted(triggered[:,i], True)
                            for i, _ in enumerate(self.thresholds)])

        # is alarm triggered at all?
        positive = triggered[-1, :]

        if reference:

            target_trial = True
            true_negative = 0
            false_positive = 0

            true_positive = positive
            false_negative = ~true_positive

            absolute_latency = np.take(absolute_latency, indices, mode='clip')
            speaker_latency = np.take(speaker_latency, indices, mode='clip')

            # the notion of "latency" is not applicable to missed detections
            absolute_latency[false_negative] = np.NAN
            speaker_latency[false_negative] = np.NAN

        else:

            target_trial = False
            true_positive = 0
            false_negative = 0

            false_positive = positive
            true_negative = ~false_positive

            # the notion of "latency" is not applicable to non-target trials
            absolute_latency = np.NAN
            speaker_latency = np.NAN

        return {
            'target': target_trial,
            'non_target': ~target_trial,
            'true_positive': true_positive,
            'true_negative': true_negative,
            'false_positive': false_positive,
            'false_negative': false_negative,
            'absolute_latency': absolute_latency,
            'speaker_latency': speaker_latency,
            'score': np.max(scores)
        }

    @property
    def absolute_latency(self):
        latencies = [trial['absolute_latency'] for _, trial in self
                                               if trial['target']]
        return np.nanmean(latencies, axis=0)

    @property
    def speaker_latency(self):
        latencies = [trial['speaker_latency'] for _, trial in self
                                              if trial['target']]
        return np.nanmean(latencies, axis=0)

    def det_curve(self, cost_miss=100, cost_fa=1, prior_target=0.01,
                  return_latency=False):
        """DET curve

        Parameters
        ----------
        cost_miss : float, optional
            Cost of missed detections. Defaults to 100.
        cost_fa : float, optional
            Cost of false alarms. Defaults to 1.
        prior_target : float, optional
            Target trial prior. Defaults to 0.5.
        return_latency : bool, optional
            Set to True to return latency.

        Returns
        -------
        thresholds : numpy array
            Detection thresholds
        fpr : numpy array
            False alarm rate
        fnr : numpy array
            False rejection rate
        eer : float
            Equal error rate
        cdet : numpy array
            Cdet cost function
        speaker_latency : numpy array
        absolute_latency : numpy array
            Speaker and absolute latency when return_latency is set to True.
        """
        y_true = np.array([trial['target'] for _, trial in self])
        scores = np.array([trial['score'] for _, trial in self])
        fpr, fnr, thresholds, eer =  det_curve(y_true, scores, distances=False)
        fpr, fnr, thresholds = fpr[::-1], fnr[::-1], thresholds[::-1]
        cdet = cost_miss * fnr * prior_target + \
               cost_fa * fpr * (1. - prior_target)

        if return_latency:
            # needed to align the thresholds used in the DET curve
            # with (self.)thresholds used to compute latencies.
            indices = np.searchsorted(thresholds, self.thresholds, side='left')

            thresholds = np.take(thresholds, indices, mode='clip')
            fpr = np.take(fpr, indices, mode='clip')
            fnr = np.take(fnr, indices, mode='clip')
            cdet = np.take(cdet, indices, mode='clip')
            return thresholds, fpr, fnr, eer, cdet, \
                   self.speaker_latency, self.absolute_latency

        else:
            return thresholds, fpr, fnr, eer, cdet

    @property
    def precision_recall_curve(self):
        """Precision-recall curve

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
        y_true = np.array([trial['target'] for _, trial in self])
        scores = np.array([trial['score'] for _, trial in self])
        return precision_recall_curve(y_true, scores, distances=False)
