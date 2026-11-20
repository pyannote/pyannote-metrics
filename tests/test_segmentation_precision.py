from pyannote.core import Segment, Timeline

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


def test_segmentation_precision_recall_uem():
    ref = Timeline([Segment(0, 1), Segment(1, 2)])
    hyp = Timeline([Segment(0, 1), Segment(1, 2), Segment(2, 3)])
    uem = Timeline([Segment(0, 2)])

    # the hypothesis boundary at 2 is on the edge of the evaluation map, so
    # only the matching boundary at 1 is evaluated
    assert SegmentationPrecision()(ref, hyp, uem=uem) == 1.0
    assert SegmentationRecall()(hyp, ref, uem=uem) == 1.0

    # without uem, every boundary is still evaluated
    assert SegmentationPrecision()(ref, hyp) == 0.5
    assert SegmentationRecall()(hyp, ref) == 0.5


def test_segmentation_precision_uem_gap():
    ref = Timeline([Segment(0, 6)])
    hyp = Timeline([Segment(0, 1), Segment(1, 6)])
    uem = Timeline([Segment(0, 2), Segment(3, 6)])

    # a gap in the evaluation map does not create boundaries on its own
    assert SegmentationPrecision()(ref, hyp, uem=uem) == 0.0
    assert SegmentationRecall()(ref, hyp, uem=uem) == 1.0
