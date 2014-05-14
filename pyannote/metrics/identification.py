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

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import unicode_literals

import numpy as np
from munkres import Munkres

from base import BaseMetric
from base import Precision, PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED
from base import Recall, RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED
from pyannote.core.annotation import Unknown

IER_TOTAL = 'total'
IER_CORRECT = 'correct'
IER_CONFUSION = 'confusion'
IER_FALSE_ALARM = 'false alarm'
IER_MISS = 'miss'
IER_NAME = 'identification error rate'


class IDMatcher(object):
    """
    ID matcher base class.

    All ID matcher classes must inherit from this class and implement
    .oneToOneMatch() -- ie return True if two IDs match and False
    otherwise.
    """

    def __init__(self):
        super(IDMatcher, self).__init__()
        self.munkres = Munkres()

    def oneToOneMatch(self, id1, id2):
        # Two IDs match if they are equal to each other
        return id1 == id2

    def manyToManyMatch(self, ids1, ids2):
        ids1 = list(ids1)
        ids2 = list(ids2)

        n1 = len(ids1)
        n2 = len(ids2)

        nCorrect = nConfusion = nMiss = nFalseAlarm = 0
        correct = list()
        confusion = list()
        miss = list()
        falseAlarm = list()

        n = max(n1, n2)

        if n > 0:

            match = np.zeros((n, n), dtype=bool)

            for i1, id1 in enumerate(ids1):
                for i2, id2 in enumerate(ids2):
                    match[i1, i2] = self.oneToOneMatch(id1, id2)

            mapping = self.munkres.compute(1-match)

            for i1, i2 in mapping:
                if i1 >= n1:
                    nFalseAlarm += 1
                    falseAlarm.append(ids2[i2])
                elif i2 >= n2:
                    nMiss += 1
                    miss.append(ids1[i1])
                elif match[i1, i2]:
                    nCorrect += 1
                    correct.append((ids1[i1], ids2[i2]))
                else:
                    nConfusion += 1
                    confusion.append((ids1[i1], ids2[i2]))

        return ({IER_CORRECT: nCorrect,
                IER_CONFUSION: nConfusion,
                IER_MISS: nMiss,
                IER_FALSE_ALARM: nFalseAlarm,
                IER_TOTAL: n1},
                {IER_CORRECT: correct,
                 IER_CONFUSION: confusion,
                 IER_MISS: miss,
                 IER_FALSE_ALARM: falseAlarm})


class UnknownIDMatcher(IDMatcher):
    """
    Two IDs match if:
    * they are both anonymous, or
    * they are both named and equal.

    """

    def __init__(self):
        super(UnknownIDMatcher, self).__init__()

    def oneToOneMatch(self, id1, id2):
        return (isinstance(id1, Unknown) and isinstance(id2, Unknown)) \
            or id1 == id2


class IdentificationErrorRate(BaseMetric):
    """


        ``ier = (wc x confusion + wf x false_alarm + wm x miss) / total``

    where
        - `confusion` is the total confusion duration in seconds
        - `false_alarm` is the total hypothesis duration where there are
        - `miss` is
        - `total` is the total duration of all tracks
        - wc, wf and wm are optional weights (default to 1)

    Parameters
    ----------
    matcher : `IDMatcher`, optional
        Defaults to `UnknownIDMatcher` instance
        i.e. two Unknowns are always considered as correct.
    unknown : bool, optional
        Set `unknown` to True (default) to take `Unknown` instances into
        account. Set it to False to get rid of them before evaluation.
    confusion, miss, false_alarm: float, optional
        Optional weights for confusion, miss and false alarm respectively.
        Default to 1. (no weight)

    """

    @classmethod
    def metric_name(cls):
        return IER_NAME

    @classmethod
    def metric_components(cls):
        return [IER_CONFUSION, IER_FALSE_ALARM, IER_MISS,
                IER_TOTAL, IER_CORRECT]

    def __init__(self,
        matcher=None, unknown=True,
        confusion=1., miss=1., false_alarm=1.,
        **kargs
    ):

        super(IdentificationErrorRate, self).__init__()
        self.matcher = UnknownIDMatcher() if matcher is None else matcher
        self.unknown = unknown
        self.confusion = confusion
        self.miss = miss
        self.false_alarm = false_alarm

    # def _get_collar(self, reference):

    #     collar = Timeline()
    #     timeline = reference.get_timeline()
    #     for segment in reference.itersegments():
    #         t = segment.start
    #         collar.add(Segment(t-self.collar, t+self.collar))
    #         t = segment.end
    #         collar.add(Segment(t-self.collar, t+self.collar))
    #     return collar.coverage()

    def _get_details(self, reference, hypothesis, **kwargs):

        detail = self._init_details()

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(hypothesis.get_timeline())
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

            counts, _ = self.matcher.manyToManyMatch(r, h)

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
            return numerator/denominator

    def _pretty(self, detail):
        string = ""
        string += "  - duration: %.2f seconds\n" % (detail[IER_TOTAL])
        string += "  - correct: %.2f seconds\n" % (detail[IER_CORRECT])
        string += "  - confusion: %.2f seconds\n" % (detail[IER_CONFUSION])
        string += "  - miss: %.2f seconds\n" % (detail[IER_MISS])
        string += "  - false alarm: %.2f seconds\n" % (detail[IER_FALSE_ALARM])
        string += "  - %s: %.2f %%\n" % (self.name, 100*detail[self.name])
        return string


class IdentificationPrecision(Precision):
    """
    Identification Precision

    Parameters
    ----------
    matcher : `IDMatcher`, optional
        Defaults to `UnknownIDMatcher` instance
    unknown : bool, optional
        Set `unknown` to True to take `Unknown` instances into account.
        Set it to False (default) to get rid of them before evaluation.
    """

    def __init__(self, matcher=None, unknown=False, **kwargs):
        super(IdentificationPrecision, self).__init__()
        self.matcher = UnknownIDMatcher if matcher is None else matcher
        self.unknown = unknown

    def _get_details(self, reference, hypothesis, **kwargs):

        detail = self._init_details()

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(hypothesis.get_timeline())
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
            detail[PRECISION_RELEVANT_RETRIEVED] += duration * counts[IER_CORRECT]

        return detail


class IdentificationRecall(Recall):
    """
    Identification Recall

    Parameters
    ----------
    matcher : `IDMatcher`, optional
        Defaults to `UnknownIDMatcher` instance
    unknown : bool, optional
        Set `unknown` to True to take `Unknown` instances into account.
        Set it to False (default) to get rid of them before evaluation.
    """

    def __init__(self, matcher=None, unknown=False, **kwargs):
        super(IdentificationRecall, self).__init__()
        self.matcher = UnknownIDMatcher if matcher is None else matcher
        self.unknown = unknown

    def _get_details(self, reference, hypothesis, **kwargs):

        detail = self._init_details()

        # common (up-sampled) timeline
        common_timeline = reference.get_timeline().union(hypothesis.get_timeline())
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


if __name__ == "__main__":
    import doctest
    doctest.testmod()
