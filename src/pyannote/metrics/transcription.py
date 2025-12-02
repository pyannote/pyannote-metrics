from typing import Callable

try:
    import meeteval
    from meeteval.io.seglst import SegLST, SegLstSegment
    from meeteval.wer.wer.orc import OrcErrorRate
    from meeteval.wer.wer.cp import CPErrorRate
except ImportError as e:
    raise ImportError(
        "The 'meeteval' package is missing. "
        "You can install it with `uv add pyannote-metrics[transcription]`"
    ) from e

from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents

TOTAL = "total"
INSERTION = "insertion"
DELETION = "deletion"
SUBSTITUTION = "substitution"


class WordErrorRate(BaseMetric):
    """Word Error Rate"""

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of the metric."""
        return "WER"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        """Return the list of metric components."""
        return [
            TOTAL,
            INSERTION,
            DELETION,
            SUBSTITUTION,
        ]

    def __init__(self, normalizer: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = normalizer or (lambda word: word)

    def _normalize(self, seglst: SegLST) -> SegLST:
        return SegLST(
            [SegLstSegment({**s, "words": self.normalizer(s["words"])}) for s in seglst]
        )

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        # check that reference is single session
        reference_session_ids = set(s["session_id"] for s in reference)
        assert (
            len(reference_session_ids) == 1
        ), "Reference must contain exactly one session"

        # keep track of that session_id
        session_id = reference_session_ids.pop()

        # check that hypothesis is for that same single session
        assert all(
            s["session_id"] == session_id for s in hypothesis
        ), f"All hypothesis segments must belong to session {session_id}"

        # normalize both reference and hypothesis
        normalized_reference: SegLST = self._normalize(reference)
        normalized_hypothesis: SegLST = self._normalize(hypothesis)

        ref_sorted = sorted(normalized_reference, key=lambda s: s["start_time"])
        hyp_sorted = sorted(normalized_hypothesis, key=lambda s: s["start_time"])

        # Build concatenated word sequences in time order
        ref_txt = " ".join(s["words"] for s in ref_sorted)
        hyp_txt = " ".join(s["words"] for s in hyp_sorted)

        # Build reference and hypothesis dicts
        ref = {
            "words": ref_txt,
        }

        hyp = {
            "words": hyp_txt,
        }

        result = meeteval.wer.siso_word_error_rate([ref], [hyp])

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }

    def compute_metric(self, detail: Details) -> float:
        numerator = detail[INSERTION] + detail[SUBSTITUTION] + detail[DELETION]
        denominator = detail[TOTAL]
        if denominator == 0.0:
            if numerator == 0:
                return 0.0
            else:
                return 1.0
        else:
            return numerator / denominator


class ConcatenatedMinimumPermutationWordErrorRate(BaseMetric):
    """Concatenated minimum-Permutation Word Error Rate (cpWER)."""

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of the metric."""
        return "Concatenated minimum-Permutation Word Error Rate"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        """Return the list of metric components."""
        return [
            TOTAL,
            INSERTION,
            DELETION,
            SUBSTITUTION,
        ]

    def __init__(self, normalizer: Callable | None = None, **kwargs):
        super().__init__(**kwargs)
        self.normalizer = normalizer or (lambda word: word)

    def _normalize(self, seglst: SegLST) -> SegLST:
        return SegLST(
            [SegLstSegment({**s, "words": self.normalizer(s["words"])}) for s in seglst]
        )

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        # check that reference is single session
        reference_session_ids = set(s["session_id"] for s in reference)
        if len(reference_session_ids) != 1:
            raise ValueError("Reference must contain exactly one session")

        # keep track of that session_id
        session_id = reference_session_ids.pop()

        # check that hypothesis is for that same single session
        if not all(s["session_id"] == session_id for s in hypothesis):
            raise ValueError(
                "All session_id values in hypothesis must match the reference session_id."
            )

        # normalize both reference and hypothesis
        normalized_reference: SegLST = self._normalize(reference)
        normalized_hypothesis: SegLST = self._normalize(hypothesis)

        # compute concatenated minimum-permutation WER
        result: CPErrorRate = meeteval.wer.cpwer(
            normalized_reference,
            normalized_hypothesis,
        )[session_id]

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }

    def compute_metric(self, detail: Details) -> float:
        numerator = detail[INSERTION] + detail[SUBSTITUTION] + detail[DELETION]
        denominator = detail[TOTAL]
        if denominator == 0.0:
            if numerator == 0:
                return 0.0
            else:
                return 1.0
        else:
            return numerator / denominator


class TimeConstrainedMinimumPermutationWordErrorRate(BaseMetric):
    """Time-Constrained minimum-Permutation Word Error Rate (tcpWER).

    Parameters
    ----------
    collar : float, optional
        Collar applied to hypothesis pseudo-word level timings, in seconds.
        Defaults to 5 seconds.
    """

    @classmethod
    def metric_name(cls) -> str:
        """Return the name of the metric."""
        return "Time-Constrained minimum-Permutation Word Error Rate"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        """Return the list of metric components."""
        return [
            TOTAL,
            INSERTION,
            DELETION,
            SUBSTITUTION,
        ]

    def __init__(
        self, normalizer: Callable | None = None, collar: float = 5.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.collar = collar
        self.normalizer = normalizer or (lambda word: word)

    def _normalize(self, seglst: SegLST) -> SegLST:
        return SegLST(
            [SegLstSegment({**s, "words": self.normalizer(s["words"])}) for s in seglst]
        )

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
    ) -> Details:
        # check that reference is single session
        reference_session_ids = set(s["session_id"] for s in reference)
        if len(reference_session_ids) != 1:
            raise ValueError("Reference must contain exactly one session")

        # keep track of that session_id
        session_id = reference_session_ids.pop()

        # check that hypothesis is for that same single session
        if not all(s["session_id"] == session_id for s in hypothesis):
            raise ValueError(
                "All session_id values in hypothesis must match the reference session_id."
            )

        # normalize both reference and hypothesis
        normalized_reference: SegLST = self._normalize(reference)
        normalized_hypothesis: SegLST = self._normalize(hypothesis)

        # compute time-constrained minimum-permutation WER
        result: CPErrorRate = meeteval.wer.tcpwer(
            normalized_reference, normalized_hypothesis, collar=self.collar
        )[session_id]

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }

    def compute_metric(self, detail: Details) -> float:
        numerator = detail[INSERTION] + detail[SUBSTITUTION] + detail[DELETION]
        denominator = detail[TOTAL]
        if denominator == 0.0:
            if numerator == 0:
                return 0.0
            else:
                return 1.0
        else:
            return numerator / denominator


class TimeConstrainedOptimalReferenceCombinationWordErrorRate(BaseMetric):
    """Time-Constrained Optimal Reference Combination Word Error Rate (tcORCWER)

    Parameters
    ----------
    collar : float, optional
        Collar applied to hypothesis pseudo-word level timings, in seconds.
        Defaults to 5 seconds.
    """

    @classmethod
    def metric_name(cls) -> str:
        return "tc-orcWER"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [
            TOTAL,
            INSERTION,
            DELETION,
            SUBSTITUTION,
        ]

    def __init__(
        self, normalizer: Callable | None = None, collar: float = 5.0, **kwargs
    ):
        super().__init__(**kwargs)
        self.normalizer = normalizer or (lambda word: word)
        self.collar = collar

    def _normalize(self, seglst: SegLST) -> SegLST:
        return SegLST(
            [SegLstSegment({**s, "words": self.normalizer(s["words"])}) for s in seglst]
        )

    def compute_components(
        self,
        reference: SegLST,
        hypothesis: SegLST,
        uem: str | None = None,
    ) -> Details:
        # check that reference is single session
        reference_session_ids = set(s["session_id"] for s in reference)
        assert len(reference_session_ids) == 1

        # keep track of that session_id
        session_id = reference_session_ids.pop()

        # check that hypothesis is for that same single session
        assert all(s["session_id"] == session_id for s in hypothesis)

        # normalize both reference and hypothesis
        normalized_reference: SegLST = self._normalize(reference)
        normalized_hypothesis: SegLST = self._normalize(hypothesis)

        # compute time-constrained minimum-permutation WER
        result: OrcErrorRate = meeteval.wer.tcorcwer(
            normalized_reference, normalized_hypothesis, collar=self.collar, uem=uem
        )[session_id]

        # keep track of components
        return {
            TOTAL: result.length,
            INSERTION: result.insertions,
            DELETION: result.deletions,
            SUBSTITUTION: result.substitutions,
        }

    def compute_metric(self, detail: Details) -> float:
        numerator = detail[INSERTION] + detail[SUBSTITUTION] + detail[DELETION]
        denominator = detail[TOTAL]
        if denominator == 0.0:
            if numerator == 0:
                return 0.0
            else:
                return 1.0
        else:
            return numerator / denominator
