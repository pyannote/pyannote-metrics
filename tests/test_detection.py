import pytest

import pyannote.core
from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.detection import DetectionPrecision
from pyannote.metrics.detection import DetectionRecall
from pyannote.metrics.detection import DetectionAccuracy
from pyannote.metrics.detection import DetectionPrecisionRecallFMeasure


import numpy.testing as npt

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# Reference   |--------------|  |-----------|     |-----|  |--------------|

# Time        0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20
# Hypothesis     |-----------------|-----|     |-----------------|  |-----|
#                                  |--------|

@pytest.fixture
def reference():
    reference = Annotation()
    reference[Segment(0, 5)] = 'A'
    reference[Segment(6, 10)] = 'B'
    reference[Segment(12, 14)] = 'A'
    reference[Segment(15, 20)] = 'C'
    return reference


@pytest.fixture
def hypothesis():
    hypothesis = Annotation()
    hypothesis[Segment(1, 7)] = 'A'
    hypothesis[Segment(7, 9)] = 'D'
    hypothesis[Segment(7, 10)] = 'B'
    hypothesis[Segment(11, 17)] = 'C'
    hypothesis[Segment(18, 20)] = 'D'
    return hypothesis


def test_error_rate(reference, hypothesis):
    detectionErrorRate = DetectionErrorRate()
    error_rate = detectionErrorRate(reference, hypothesis)
    npt.assert_almost_equal(error_rate, 0.3125, decimal=7)


def test_detailed(reference, hypothesis):
    detectionErrorRate = DetectionErrorRate()
    details = detectionErrorRate(reference, hypothesis, detailed=True)

    rate = details['detection error rate']
    npt.assert_almost_equal(rate, 0.3125, decimal=7)

    false_alarm = details['false alarm']
    npt.assert_almost_equal(false_alarm, 3.0, decimal=7)

    missed_detection = details['miss']
    npt.assert_almost_equal(missed_detection, 2.0, decimal=7)

    total = details['total']
    npt.assert_almost_equal(total, 16.0, decimal=7)


def test_accuracy(reference, hypothesis):
    # 15 correct / 20 total
    detectionAccuracy = DetectionAccuracy()
    accuracy = detectionAccuracy(reference, hypothesis)
    npt.assert_almost_equal(accuracy, 0.75, decimal=3)


def test_precision(reference, hypothesis):
    # 14 true positive / 17 detected
    detectionPrecision = DetectionPrecision()
    precision = detectionPrecision(reference, hypothesis)
    npt.assert_almost_equal(precision, 0.8235, decimal=3)


def test_recall(reference, hypothesis):
    # 14 true positive / 16 expected
    detectionRecall = DetectionRecall()
    recall = detectionRecall(reference, hypothesis)
    npt.assert_almost_equal(recall, 0.875, decimal=3)


def test_fscore(reference, hypothesis):
    # expected 28/33 since it
    # is computed as :
    # 2*precision*recall / (precision+recall)
    detectionFMeasure = DetectionPrecisionRecallFMeasure()
    precision, recall, fscore = detectionFMeasure(reference, hypothesis)
    npt.assert_almost_equal([precision, recall, fscore], [0.8235, 0.875, 0.848], decimal=3)
