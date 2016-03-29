#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2014 CNRS

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

from .base import BaseMetric
from .base import Precision, PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED
from .base import Recall, RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED

from .utils import UEMSupportMixin

DER_TOTAL = 'total'
DER_FALSE_ALARM = 'false alarm'
DER_MISS = 'miss'
DER_NAME = 'detection error rate'


class DetectionErrorRate(UEMSupportMixin, BaseMetric):
    """Detection error rate

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments
    """

    @classmethod
    def metric_name(cls):
        return DER_NAME

    @classmethod
    def metric_components(cls):
        return [DER_FALSE_ALARM, DER_MISS, DER_TOTAL]

    def __init__(self, collar=0., **kargs):

        super(DetectionErrorRate, self).__init__()
        self.collar = collar

    def _get_details(self, reference, hypothesis, uem=None, **kwargs):

        detail = self._init_details()

        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem, collar=self.collar)

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(
            hypothesis.get_timeline())
        common_timeline = common_timeline.segmentation()

        # align reference on common timeline
        R = self._tagger(reference, common_timeline)

        # translate and align hypothesis on common timeline
        H = self._tagger(hypothesis, common_timeline)

        # loop on all segments
        for segment in common_timeline:

            # segment duration
            duration = segment.duration

            # set of IDs in reference segment
            r = R.get_labels(segment)
            Nr = len(r)
            detail[DER_TOTAL] += duration * Nr

            # set of IDs in hypothesis segment
            h = H.get_labels(segment)
            Nh = len(h)

            # number of misses
            N_miss = max(0, Nr - Nh)
            detail[DER_MISS] += duration * N_miss

            # number of false alarms
            N_fa = max(0, Nh - Nr)
            detail[DER_FALSE_ALARM] += duration * N_fa

        return detail

    def _get_rate(self, detail):
        numerator = 1. * (detail[DER_FALSE_ALARM] + detail[DER_MISS])
        denominator = 1. * detail[DER_TOTAL]
        if denominator == 0.:
            if numerator == 0:
                return 0.
            else:
                return 1.
        else:
            return numerator / denominator

    def _pretty(self, detail):
        string = ""
        string += "  - duration: %.2f seconds\n" % (detail[DER_TOTAL])
        string += "  - miss: %.2f seconds\n" % (detail[DER_MISS])
        string += "  - false alarm: %.2f seconds\n" % (detail[DER_FALSE_ALARM])
        string += "  - %s: %.2f %%\n" % (self.name, 100 * detail[self.name])
        return string


class DetectionPrecision(UEMSupportMixin, Precision):
    """Detection precision

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments
    """
    def __init__(self, collar=0., **kargs):
        super(DetectionPrecision, self).__init__()
        self.collar = collar

    def _get_details(self, reference, hypothesis, uem=None, **kwargs):

        detail = self._init_details()

        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem, collar=self.collar)

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(
            hypothesis.get_timeline())
        common_timeline = common_timeline.segmentation()

        # align reference on common timeline
        R = self._tagger(reference, common_timeline)

        # translate and align hypothesis on common timeline
        H = self._tagger(hypothesis, common_timeline)

        # loop on all segments
        for segment in common_timeline:

            # segment duration
            duration = segment.duration

            # set of IDs in reference segment
            r = R.get_labels(segment)
            Nr = len(r)

            # set of IDs in hypothesis segment
            h = H.get_labels(segment)
            Nh = len(h)

            detail[PRECISION_RETRIEVED] += duration * Nh
            detail[PRECISION_RELEVANT_RETRIEVED] += duration * min(Nr, Nh)

        return detail


class DetectionRecall(UEMSupportMixin, Recall):
    """Detection recall

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments
    """
    def __init__(self, collar=0., **kargs):
        super(DetectionRecall, self).__init__()
        self.collar = collar

    def _get_details(self, reference, hypothesis, uem=None, **kwargs):

        detail = self._init_details()

        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem, collar=self.collar)

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(
            hypothesis.get_timeline())
        common_timeline = common_timeline.segmentation()

        # align reference on common timeline
        R = self._tagger(reference, common_timeline)

        # translate and align hypothesis on common timeline
        H = self._tagger(hypothesis, common_timeline)

        # loop on all segments
        for segment in common_timeline:

            # segment duration
            duration = segment.duration

            # set of IDs in reference segment
            r = R.get_labels(segment)
            Nr = len(r)

            # set of IDs in hypothesis segment
            h = H.get_labels(segment)
            Nh = len(h)

            detail[RECALL_RELEVANT] += duration * Nr
            detail[RECALL_RELEVANT_RETRIEVED] += duration * min(Nr, Nh)

        return detail


if __name__ == "__main__":
    import doctest
    doctest.testmod()
