import pytest

import pyannote.core
from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.core import Timeline
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import DiarizationPurity
from pyannote.metrics.diarization import DiarizationCoverage

import numpy.testing as npt


@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def reference_with_overlap():
    reference = Annotation()
    reference[Segment(0, 13)] = "A"
    reference[Segment(12, 20)] = "B"
    reference[Segment(24, 27)] = "A"
    reference[Segment(30, 40)] = "C"
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(2, 13)] = "a"
    hypothesis[Segment(13, 14)] = "d"
    hypothesis[Segment(14, 20)] = "b"
    hypothesis[Segment(22, 38)] = "c"
    hypothesis[Segment(38, 40)] = "d"
    return hypothesis


def test_error_rate(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    error_rate = diarizationErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.5161290322580645, decimal=7)


def test_optimal_mapping(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    mapping = diarizationErrorRate.optimal_mapping(reference, hypothesis)
    assert mapping == {"a": "A", "b": "B", "c": "C"}


def test_detailed(reference, hypothesis):
    diarizationErrorRate = DiarizationErrorRate()
    details = diarizationErrorRate(reference, hypothesis, detailed=True)

    confusion = details["confusion"]
    npt.assert_almost_equal(confusion, 7.0, decimal=7)

    correct = details["correct"]
    npt.assert_almost_equal(correct, 22.0, decimal=7)

    rate = details["diarization error rate"]
    npt.assert_almost_equal(rate, 0.5161290322580645, decimal=7)

    false_alarm = details["false alarm"]
    npt.assert_almost_equal(false_alarm, 7.0, decimal=7)

    missed_detection = details["missed detection"]
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details["total"]
    npt.assert_almost_equal(total, 31.0, decimal=7)


def test_purity(reference, hypothesis):
    diarizationPurity = DiarizationPurity()
    purity = diarizationPurity(reference, hypothesis)
    npt.assert_almost_equal(purity, 0.6666, decimal=3)


def test_coverage(reference, hypothesis):
    diarizationCoverage = DiarizationCoverage()
    coverage = diarizationCoverage(reference, hypothesis)
    npt.assert_almost_equal(coverage, 0.7096, decimal=3)


def test_skip_overlap(reference_with_overlap, hypothesis):
    metric = DiarizationErrorRate(skip_overlap=True)
    total = metric(reference_with_overlap, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 32, decimal=3)


def test_leep_overlap(reference_with_overlap, hypothesis):
    metric = DiarizationErrorRate(skip_overlap=False)
    total = metric(reference_with_overlap, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 34, decimal=3)


def test_bug_16():
    reference = Annotation()
    reference[Segment(0, 10)] = "A"
    hypothesis = Annotation()

    metric = DiarizationErrorRate(collar=1)
    total = metric(reference, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 9, decimal=3)

    metric = DiarizationErrorRate(collar=0)
    total = metric(reference, hypothesis, detailed=True)["total"]
    npt.assert_almost_equal(total, 10, decimal=3)
