from pyannote.core import Segment, Timeline

from pyannote.metrics.segmentation import SegmentationPrecision, SegmentationRecall


def test_segmentation_precision_recall_dense_boundaries():
    # reference boundaries are {10, 11.5}, hypothesis boundaries are {11, 12.5}
    ref = Timeline([Segment(0, 10), Segment(10, 11.5), Segment(11.5, 20)])
    hyp = Timeline([Segment(0, 11), Segment(11, 12.5), Segment(12.5, 20)])

    # 11 <-> 10 and 12.5 <-> 11.5 are both within tolerance, so every boundary
    # can be matched. matching greedily picked the closest pair (11 <-> 11.5)
    # first, which left 12.5 without a partner and halved both scores.
    assert SegmentationPrecision(tolerance=1.1)(ref, hyp) == 1.0
    assert SegmentationRecall(tolerance=1.1)(ref, hyp) == 1.0


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
