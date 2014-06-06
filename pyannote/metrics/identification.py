#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2014 CNRS (Hervé BREDIN - http://herve.niderb.fr)

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

from __future__ import unicode_literals

from base import BaseMetric
from base import Precision, PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED
from base import Recall, RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED
from matcher import LabelMatcherWithUnknownSupport, \
    MATCH_TOTAL, MATCH_CORRECT, MATCH_CONFUSION, \
    MATCH_MISSED_DETECTION, MATCH_FALSE_ALARM
from pyannote.core import Segment, Timeline

IER_TOTAL = MATCH_TOTAL
IER_CORRECT = MATCH_CORRECT
IER_CONFUSION = MATCH_CONFUSION
IER_FALSE_ALARM = MATCH_FALSE_ALARM
IER_MISS = MATCH_MISSED_DETECTION
IER_NAME = 'identification error rate'


class UEMSupportMixin:
    """Provides 'uemify' method with optional (à la NIST) collar"""

    def _get_collar(self, reference, duration):

        # initialize empty timeline
        collar = Timeline(uri=reference.uri)

        if duration == 0.:
            return collar

        # iterate over all segments in reference
        for segment in reference.itersegments():

            # add collar centered on start time
            t = segment.start
            collar.add(Segment(t - .5 * duration, t + .5 * duration))

            # add collar centered on end time
            t = segment.end
            collar.add(Segment(t - .5 * duration, t + .5 * duration))

        # merge overlapping collars and return
        return collar.coverage()

    def uemify(self, reference, hypothesis, uem=None, collar=0.):
        """

        Parameters
        ----------
        reference, hypothesis : Annotation
            Reference and hypothesis annotations.
        uem : Timeline, optional
            Evaluation map.
        collar : float, optional
            Duration (in seconds) of collars removed from evaluation around
            boundaries of reference segments.
        """

        # when uem is not provided
        # use the union of reference and hypothesis extents
        if uem is None:
            r_extent = reference.get_timeline().extent()
            h_extent = hypothesis.get_timeline().extent()
            uem = Timeline(segments=[r_extent | h_extent], uri=reference.uri)

        # remove collars from uem
        uem = self._get_collar(reference, collar).gaps(focus=uem)

        # crop reference and hypothesis
        reference = reference.crop(uem, mode='intersection')
        hypothesis = hypothesis.crop(uem, mode='intersection')

        return reference, hypothesis


class IdentificationErrorRate(UEMSupportMixin, BaseMetric):
    """Identification error rate

    ``ier = (wc x confusion + wf x false_alarm + wm x miss) / total``

    where
        - `confusion` is the total confusion duration in seconds
        - `false_alarm` is the total hypothesis duration where there are
        - `miss` is
        - `total` is the total duration of all tracks
        - wc, wf and wm are optional weights (default to 1)

    Parameters
    ----------
    matcher : `Matcher`, optional
        Defaults to `LabelMatcherWithUnknownSupport` instance
        i.e. two Unknowns are always considered as correct.
    unknown : bool, optional
        Set `unknown` to True (default) to take `Unknown` instances into
        account. Set it to False to get rid of them before evaluation.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    confusion, miss, false_alarm: float, optional
        Optional weights for confusion, miss and false alarm respectively.
        Default to 1. (no weight)
    """

    @classmethod
    def metric_name(cls):
        return IER_NAME

    @classmethod
    def metric_components(cls):
        return [
            IER_CONFUSION,
            IER_FALSE_ALARM, IER_MISS,
            IER_TOTAL, IER_CORRECT
        ]

    def __init__(self, matcher=None, unknown=True,
                 confusion=1., miss=1., false_alarm=1.,
                 collar=0., **kargs):

        super(IdentificationErrorRate, self).__init__()

        if matcher is None:
            matcher = LabelMatcherWithUnknownSupport()
        self.matcher = matcher
        self.unknown = unknown
        self.confusion = confusion
        self.miss = miss
        self.false_alarm = false_alarm
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

            # list of IDs in reference segment
            r = R.get_labels(segment, unknown=self.unknown, unique=False)

            # list of IDs in hypothesis segment
            h = H.get_labels(segment, unknown=self.unknown, unique=False)

            counts, _ = self.matcher(r, h)

            detail[IER_TOTAL] += duration * counts[IER_TOTAL]
            detail[IER_CORRECT] += duration * counts[IER_CORRECT]
            detail[IER_CONFUSION] += duration * counts[IER_CONFUSION]
            detail[IER_MISS] += duration * counts[IER_MISS]
            detail[IER_FALSE_ALARM] += duration * counts[IER_FALSE_ALARM]

        return detail

    def _get_rate(self, detail):

        numerator = 1. * (
            self.confusion * detail[IER_CONFUSION] +
            self.false_alarm * detail[IER_FALSE_ALARM] +
            self.miss * detail[IER_MISS]
        )
        denominator = 1. * detail[IER_TOTAL]
        if denominator == 0.:
            if numerator == 0:
                return 0.
            else:
                return 1.
        else:
            return numerator / denominator

    def _pretty(self, detail):
        string = ""
        string += "  - duration: %.2f seconds\n" % (detail[IER_TOTAL])
        string += "  - correct: %.2f seconds\n" % (detail[IER_CORRECT])
        string += "  - confusion: %.2f seconds\n" % (detail[IER_CONFUSION])
        string += "  - miss: %.2f seconds\n" % (detail[IER_MISS])
        string += "  - false alarm: %.2f seconds\n" % (detail[IER_FALSE_ALARM])
        string += "  - %s: %.2f %%\n" % (self.name, 100 * detail[self.name])
        return string


class IdentificationPrecision(UEMSupportMixin, Precision):
    """Identification Precision

    Parameters
    ----------
    matcher : `Matcher`, optional
        Defaults to `LabelMatcherWithUnknownSupport` instance
        i.e. two Unknowns are always considered as correct.
    unknown : bool, optional
        Set `unknown` to True (default) to take `Unknown` instances into
        account. Set it to False to get rid of them before evaluation.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    """

    def __init__(self, matcher=None, unknown=False, collar=0., **kwargs):
        super(IdentificationPrecision, self).__init__()
        if matcher is None:
            matcher = LabelMatcherWithUnknownSupport()
        self.matcher = matcher
        self.unknown = unknown
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

        # align hypothesis on common timeline
        H = self._tagger(hypothesis, common_timeline)

        # loop on all segments
        for segment in common_timeline:

            # segment duration
            duration = segment.duration

            # list of IDs in reference segment
            r = R.get_labels(segment, unknown=self.unknown, unique=False)

            # list of IDs in hypothesis segment
            h = H.get_labels(segment, unknown=self.unknown, unique=False)

            counts, _ = self.matcher.manyToManyMatch(r, h)

            detail[PRECISION_RETRIEVED] += duration * len(h)
            detail[PRECISION_RELEVANT_RETRIEVED] += \
                duration * counts[IER_CORRECT]

        return detail


class IdentificationRecall(UEMSupportMixin, Recall):
    """Identification Recall

    Parameters
    ----------
    matcher : `Matcher`, optional
        Defaults to `LabelMatcherWithUnknownSupport` instance
        i.e. two Unknowns are always considered as correct.
    unknown : bool, optional
        Set `unknown` to True (default) to take `Unknown` instances into
        account. Set it to False to get rid of them before evaluation.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    """

    def __init__(self, matcher=None, unknown=False, collar=0., **kwargs):
        super(IdentificationRecall, self).__init__()
        if matcher is None:
            matcher = LabelMatcherWithUnknownSupport()
        self.matcher = matcher
        self.unknown = unknown
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

        # align hypothesis on common timeline
        H = self._tagger(hypothesis, common_timeline)

        # loop on all segments
        for segment in common_timeline:

            # segment duration
            duration = segment.duration

            # list of IDs in reference segment
            r = R.get_labels(segment, unknown=self.unknown, unique=False)

            # list of IDs in hypothesis segment
            h = H.get_labels(segment, unknown=self.unknown, unique=False)

            counts, _ = self.matcher.manyToManyMatch(r, h)

            detail[RECALL_RELEVANT] += duration * counts[IER_TOTAL]
            detail[RECALL_RELEVANT_RETRIEVED] += duration * counts[IER_CORRECT]

        return detail
