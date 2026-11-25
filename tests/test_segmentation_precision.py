from pyannote.core import Annotation, Segment, Timeline

from pyannote.metrics.segmentation import SegmentationPrecision, SegmentationRecall


def test_segmentation_precision_empty_hypothesis():
    ref = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    # empty hypothesis with >=2 reference segments used to crash np.zeros((N, -1))
    assert SegmentationPrecision()(ref, Timeline()) == 1.0

    # accumulating an empty hypothesis used to push a boundary count of -1,
    # giving a corpus precision > 1
    m = SegmentationPrecision()
    m(ref, ref)
    m(Timeline([Segment(0, 3)]), Timeline())
    assert abs(m) <= 1.0


def test_segmentation_evaluation_map():
    ref = Timeline([Segment(0, 1), Segment(1, 2)])
    hyp = Annotation()
    for i in range(3):
        hyp[Segment(i, i + 1)] = i
    uem = Timeline([Segment(0, 1), Segment(1, 2)])
    for metric, reference, hypothesis in (
        (SegmentationPrecision(), ref, hyp),
        (SegmentationRecall(), hyp, ref),
    ):
        assert metric(reference, hypothesis) == 0.5
        assert metric(reference, hypothesis, uem=uem) == 1.0
        empty = metric(reference, hypothesis, uem=Timeline(), detailed=True)
        assert empty["number of boundaries"] == empty["number of matches"] == 0


def test_segmentation_does_not_match_across_unscored_gap():
    ref = Timeline([Segment(0, 0.9), Segment(0.9, 2)])
    hyp = Timeline([Segment(0, 1.1), Segment(1.1, 2)])
    uem = Timeline([Segment(0, 0.95), Segment(1.05, 2)])
    for metric in (SegmentationPrecision(tolerance=0.3), SegmentationRecall(tolerance=0.3)):
        assert metric(ref, hyp) == 1.0
        assert metric(ref, hyp, uem=uem) == 0.0
