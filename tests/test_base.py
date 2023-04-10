#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2020 CNRS

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


import pytest

from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.metrics.detection import DetectionAccuracy


import numpy.testing as npt

# rec1
#
# Time        0  1  2  3  4  5  6
# Reference   |-----|
# Hypothesis     |-----|
# UEM         |-----------------|

# rec2
#
# Time        0  1  2  3  4  5  6  7
# Reference      |--------|
# Hypothesis           |-----|
# UEM         |--------------------|


@pytest.fixture
def reference():
    reference = {}
    reference['rec1'] = Annotation()
    reference['rec1'][Segment(0, 2)] = 'A'
    reference['rec2'] = Annotation()
    reference['rec2'][Segment(1, 4)] = 'A'
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = {}
    hypothesis['rec1'] = Annotation()
    hypothesis['rec1'][Segment(1, 3)] = 'A'
    hypothesis['rec2'] = Annotation()
    hypothesis['rec2'][Segment(3, 4)] = 'A'
    return hypothesis


@pytest.fixture
def uem():
    return {
        'rec1': Timeline([Segment(0, 6)]),
        'rec2': Timeline([Segment(0, 7)])}


def test_summation(reference, hypothesis, uem):
    # Expected error rate.
    expected = 9 / 13

    # __add__
    m1 = DetectionAccuracy()
    m1(reference['rec1'], hypothesis['rec1'], uem=uem['rec1'])
    m2 = DetectionAccuracy()
    m2(reference['rec2'], hypothesis['rec2'], uem=uem['rec2'])
    npt.assert_almost_equal(abs(m1 + m2), expected, decimal=3)

    # __radd__
    m = sum([m1, m2])
    npt.assert_almost_equal(abs(m), expected, decimal=3)
