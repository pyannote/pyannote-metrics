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

import numpy as np
from base import BaseMetric
from pyannote.core import Annotation

PURITY_NAME = 'segmentation purity'
COVERAGE_NAME = 'segmentation coverage'
PTY_CVG_TOTAL = 'total duration'
PTY_CVG_INTER = 'intersection duration'

PK_NAME = 'segmentation pk'
WINDOWDIFF_NAME = 'segmentation windowdiff'
PK_AGREEMENT = 'number of agreements'
PK_COMPARISON = 'number of comparisons'


class SegmentationCoverage(BaseMetric):
    """Segmentation coverage

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationCoverage
    >>> cvg = SegmentationCoverage()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> cvg(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> cvg(reference, hypothesis)
    0.75
    """

    @classmethod
    def metric_name(cls):
        return COVERAGE_NAME

    @classmethod
    def metric_components(cls):
        return [PTY_CVG_TOTAL, PTY_CVG_INTER]

    def _get_details(self, reference, hypothesis, **kwargs):

        if isinstance(reference, Annotation):
            reference = reference.get_timeline()

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        detail = self._init_details()

        prev_r = None
        duration = 0.
        intersection = 0.
        for r, h in reference.co_iter(hypothesis):

            if r != prev_r:
                detail[PTY_CVG_TOTAL] += duration
                detail[PTY_CVG_INTER] += intersection

                duration = r.duration
                intersection = 0.
                prev_r = r

            intersection = max(intersection, (r & h).duration)

        detail[PTY_CVG_TOTAL] += duration
        detail[PTY_CVG_INTER] += intersection

        return detail

    def _get_rate(self, detail):

        return detail[PTY_CVG_INTER] / detail[PTY_CVG_TOTAL]

    def _pretty(self, detail):
        string = ""
        string += "  - duration: %.2f seconds\n" % (detail[PTY_CVG_TOTAL])
        string += "  - correct: %.2f seconds\n" % (detail[PTY_CVG_INTER])
        string += "  - %s: %.2f %%\n" % (self.name, 100 * detail[self.name])
        return string


class SegmentationPurity(SegmentationCoverage):
    """Segmentation purity

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationPurity
    >>> pty = SegmentationPurity()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 1))
    >>> hypothesis.add(Segment(1, 2))
    >>> hypothesis.add(Segment(2, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> pty(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> pty(reference, hypothesis)
    0.5

    """

    @classmethod
    def metric_name(cls):
        return PURITY_NAME

    def _get_details(self, reference, hypothesis, **kwargs):
        return super(SegmentationPurity, self)._get_details(
            hypothesis, reference, **kwargs
        )


class SegmentationPK(BaseMetric):
    """Segmentation PK

    Parameters
    ----------
    step : float, optional
        Step in seconds. Defaults to 1.

    Example
    -------

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationPK
    >>> pk = SegmentationPK()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> pk(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> pk(reference, hypothesis)
    0.75
    """

    def __init__(self, step=1):

        super(SegmentationPK, self).__init__()
        self.step = step

    @classmethod
    def metric_name(cls):
        return PK_NAME

    @classmethod
    def metric_components(cls):
        return [PK_AGREEMENT, PK_COMPARISON]

    def _get_details(self, reference, hypothesis, **kwargs):

        if isinstance(reference, Annotation):
            reference = reference.get_timeline()

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        detail = self._init_details()

        # (half-)average duration of hypothesis segments
        K = 0.5 * (hypothesis.duration() / len(hypothesis))

        comparisons = 0.
        agreements = 0.

        start, end = hypothesis.extent()
        for t in np.arange(start, end - K, self.step):

            # true if t and t+K fall in the same segment
            # false if t and t+K fall into 2 different segments
            r = reference.overlapping(t) == reference.overlapping(t + K)
            h = hypothesis.overlapping(t) == hypothesis.overlapping(t + K)

            # increment if reference and hypothesis agree
            agreements += (r == h)

            comparisons += 1

        detail[PK_AGREEMENT] += agreements
        detail[PK_COMPARISON] += comparisons

        return detail

    def _get_rate(self, detail):

        return 1. * detail[PK_AGREEMENT] / detail[PK_COMPARISON]

    def _pretty(self, detail):
        string = ""
        string += "  - number of comparisons: %d\n" % (detail[PK_COMPARISON])
        string += "  - number of agreements: %d\n" % (detail[PK_AGREEMENT])
        string += "  - %s: %.2f %%\n" % (self.name, 100 * detail[self.name])
        return string


class SegmentationWindowDiff(BaseMetric):
    """Segmentation WindowDiff

    Parameters
    ----------
    step : float, optional
        Step in seconds. Defaults to 1.

    Example
    -------

    >>> from pyannote.core import Timeline, Segment
    >>> from pyannote.metrics.segmentation import SegmentationWindowDiff
    >>> wd = SegmentationWindowDiff()

    >>> reference = Timeline()
    >>> reference.add(Segment(0, 1))
    >>> reference.add(Segment(1, 2))
    >>> reference.add(Segment(2, 4))

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 4))
    >>> wd(reference, hypothesis)
    1.0

    >>> hypothesis = Timeline()
    >>> hypothesis.add(Segment(0, 3))
    >>> hypothesis.add(Segment(3, 4))
    >>> wd(reference, hypothesis)
    0.75
    """

    def __init__(self, step=1):
        super(SegmentationWindowDiff, self).__init__()
        self.step = step

    @classmethod
    def metric_name(cls):
        return WINDOWDIFF_NAME

    @classmethod
    def metric_components(cls):
        return [PK_AGREEMENT, PK_COMPARISON]

    def _get_details(self, reference, hypothesis, **kwargs):

        if isinstance(reference, Annotation):
            reference = reference.get_timeline()

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline()

        detail = self._init_details()

        # (half-)average duration of hypothesis segments
        K = .5 * hypothesis.duration() / len(hypothesis)

        comparisons = 0.
        agreements = 0.

        start, end = hypothesis.extent()
        for t in np.arange(start, end - K, self.step):

            # number of boundaries between t and t+K in reference
            i = reference.index(reference.overlapping(t + K)[0])
            j = reference.index(reference.overlapping(t)[0])
            r = i - j

            # number of boundaries between t and t+K in hypothesis
            i = hypothesis.index(hypothesis.overlapping(t + K)[0])
            j = hypothesis.index(hypothesis.overlapping(t)[0])
            h = i - j

            # increment if reference and hypothesis agree
            agreements += (r == h)

            comparisons += 1

        detail[PK_AGREEMENT] += agreements
        detail[PK_COMPARISON] += comparisons

        return detail

    def _get_rate(self, detail):

        return 1. * detail[PK_AGREEMENT] / detail[PK_COMPARISON]

    def _pretty(self, detail):
        string = ""
        string += "  - number of comparisons: %d\n" % (detail[PK_COMPARISON])
        string += "  - number of agreements: %d\n" % (detail[PK_AGREEMENT])
        string += "  - %s: %.2f %%\n" % (self.name, 100 * detail[self.name])
        return string


if __name__ == "__main__":
    import doctest
    doctest.testmod()
