import pickle

import pytest

from pyannote.core import Annotation
from pyannote.core import Segment
from pyannote.metrics.detection import DetectionErrorRate
from pyannote.metrics.diarization import DiarizationErrorRate
from pyannote.metrics.diarization import JaccardErrorRate

import numpy.testing as npt


@pytest.fixture
def files():
    reference1 = Annotation(uri="file1")
    reference1[Segment(0, 10)] = "A"
    reference1[Segment(12, 20)] = "B"
    reference1[Segment(24, 27)] = "A"
    reference1[Segment(30, 40)] = "C"

    hypothesis1 = Annotation(uri="file1")
    hypothesis1[Segment(2, 13)] = "a"
    hypothesis1[Segment(13, 14)] = "d"
    hypothesis1[Segment(14, 20)] = "b"
    hypothesis1[Segment(22, 38)] = "c"
    hypothesis1[Segment(38, 40)] = "d"

    reference2 = Annotation(uri="file2")
    reference2[Segment(0, 5)] = "A"
    reference2[Segment(6, 10)] = "B"
    reference2[Segment(12, 13)] = "B"
    reference2[Segment(15, 20)] = "A"

    hypothesis2 = Annotation(uri="file2")
    hypothesis2[Segment(1, 6)] = "a"
    hypothesis2[Segment(6, 7)] = "b"
    hypothesis2[Segment(7, 10)] = "c"
    hypothesis2[Segment(11, 19)] = "b"
    hypothesis2[Segment(19, 20)] = "a"

    return [(reference1, hypothesis1), (reference2, hypothesis2)]


def evaluate_separately(make_metric, files):
    metrics = []
    for reference, hypothesis in files:
        metric = make_metric()
        metric(reference, hypothesis)
        metrics.append(metric)
    return metrics


@pytest.mark.parametrize(
    "make_metric",
    [
        lambda: DiarizationErrorRate(collar=0.5, skip_overlap=True),
        lambda: DetectionErrorRate(collar=0.5),
    ],
)
def test_sum_matches_single_metric(files, make_metric):
    single = make_metric()
    for reference, hypothesis in files:
        single(reference, hypothesis)

    per_file = evaluate_separately(make_metric, files)
    # otherwise the test could not tell summing apart from keeping one file
    assert abs(per_file[0]) != pytest.approx(abs(single))

    for combined in (per_file[0] + per_file[1], sum(per_file)):
        npt.assert_almost_equal(abs(combined), abs(single), decimal=7)
        assert combined[:] == pytest.approx(single[:])
        assert [uri for uri, _ in combined] == ["file1", "file2"]


def test_sum_after_pickling(files):
    # joblib and multiprocessing send each worker's metric back pickled
    single = DiarizationErrorRate(collar=0.5)
    for reference, hypothesis in files:
        single(reference, hypothesis)

    per_file = evaluate_separately(lambda: DiarizationErrorRate(collar=0.5), files)
    per_file = [pickle.loads(pickle.dumps(metric)) for metric in per_file]

    npt.assert_almost_equal(abs(sum(per_file)), abs(single), decimal=7)


def test_clone_keeps_options_but_not_results(files):
    reference, hypothesis = files[0]
    metric = DiarizationErrorRate(collar=0.5, skip_overlap=True)
    metric(reference, hypothesis)

    clone = metric.clone()

    assert clone is not metric
    assert (clone.collar, clone.skip_overlap) == (0.5, True)
    assert clone.results_ == []
    assert all(value == 0.0 for value in clone.accumulated_.values())
    assert len(metric.results_) == 1


def test_add_refuses_different_options(files):
    reference, hypothesis = files[0]
    loose = DiarizationErrorRate(collar=0.5)
    strict = DiarizationErrorRate(collar=0.0)
    loose(reference, hypothesis)
    strict(reference, hypothesis)

    with pytest.raises(ValueError, match="collar"):
        loose + strict


def test_add_refuses_different_metrics(files):
    reference, hypothesis = files[0]
    der = DiarizationErrorRate()
    jer = JaccardErrorRate()
    der(reference, hypothesis)
    jer(reference, hypothesis)

    with pytest.raises(TypeError, match="Cannot add"):
        der + jer


def test_sum_does_not_modify_summands(files):
    per_file = evaluate_separately(DiarizationErrorRate, files)
    accumulated = [dict(metric.accumulated_) for metric in per_file]

    total = sum(per_file)
    total(*files[0])

    assert [metric.accumulated_ for metric in per_file] == accumulated
    assert [len(metric.results_) for metric in per_file] == [1, 1]
