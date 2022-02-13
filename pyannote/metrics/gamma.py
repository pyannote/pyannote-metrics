import os
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Union, Optional, Tuple

import numpy as np
from pyannote.core import Annotation, Timeline
try:
    from pygamma_agreement import GammaResults, PositionalSporadicDissimilarity, Continuum, \
        CombinedCategoricalDissimilarity, AbsoluteCategoricalDissimilarity, \
        AbstractDissimilarity
    from pygamma_agreement.continuum import _compute_gamma_k_job
except ImportError as err:
    raise ImportError("pygamma-agreement cannot be imported, "
                      "run `pip install pyannote.metrics[gamma]` "
                      "to fix this") from err

from sortedcontainers import SortedSet

from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents
from pyannote.metrics.utils import UEMSupportMixin

__all__ = [
    'GammaDetectionError',
    'GammaIdentificationError',
    'GammaCategorizationError'
]

GAMMA_DISORDER = "disorder"
GAMMA_CHANCE_DISORDER = "chance disorder"


class BaseGammaMetric(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls) -> str:
        return "BaseGamma"

    dissim: AbstractDissimilarity

    def __init__(self,
                 collar: float = 0.,
                 skip_overlap: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [GAMMA_DISORDER, GAMMA_CHANCE_DISORDER]

    def compute_metric(self, components: Details):
        # TODO: add boundary check when expected_disorder == 0.0
        return 1 - (components[GAMMA_DISORDER] / components[GAMMA_CHANCE_DISORDER])

    def compute_gamma(self,
                      reference: Union[Annotation, Timeline],
                      hypothesis: Union[Annotation, Timeline],
                      ) -> Tuple[float, float]:
        raise NotImplementedError()

    def compute_components(self,
                           reference: Union[Annotation, Timeline],
                           hypothesis: Union[Annotation, Timeline],
                           uem: Optional[Timeline] = None,
                           collar: Optional[float] = None,
                           skip_overlap: Optional[float] = None,
                           **kwargs) -> Details:

        if collar is None:
            collar = self.collar
        if skip_overlap is None:
            skip_overlap = self.skip_overlap

        reference, hypothesis, uem = self.uemify(
            reference, hypothesis, uem=uem,
            collar=collar, skip_overlap=skip_overlap,
            returns_uem=True)

        observed, expected = self.compute_gamma(reference, hypothesis)
        return {
            GAMMA_DISORDER: observed,
            GAMMA_CHANCE_DISORDER: expected
        }


class GammaDetectionError(BaseGammaMetric, UEMSupportMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dissim = PositionalSporadicDissimilarity(delta_empty=1.0)

    @classmethod
    def metric_name(cls) -> str:
        return "GammaDet"

    def compute_gamma(self,
                      reference: Union[Annotation, Timeline],
                      hypothesis: Union[Annotation, Timeline],
                      ) -> Tuple[float, float]:
        if isinstance(reference, Annotation):
            reference = reference.get_timeline(copy=True)

        if isinstance(hypothesis, Annotation):
            hypothesis = hypothesis.get_timeline(copy=True)

        continuum = Continuum()
        continuum.add_timeline("reference", reference)
        continuum.add_timeline("hypothesis", hypothesis)
        gamma_results = continuum.compute_gamma(self.dissim,
                                                precision_level="medium",
                                                ground_truth_annotators=SortedSet(["reference"]),
                                                soft=True)
        return gamma_results.observed_disorder, gamma_results.observed_disorder


class GammaIdentificationError(BaseGammaMetric, UEMSupportMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dissim = CombinedCategoricalDissimilarity(
            pos_dissim=PositionalSporadicDissimilarity(delta_empty=1.0),
            cat_dissim=AbsoluteCategoricalDissimilarity(delta_empty=1.0)
        )

    @classmethod
    def metric_name(cls) -> str:
        return "GammaId"

    def compute_gamma(self,
                      reference: Annotation,
                      hypothesis: Annotation,
                      ) -> Tuple[float, float]:
        assert isinstance(reference, Annotation)
        assert isinstance(hypothesis, Annotation)
        continuum = Continuum()
        continuum.add_annotation("reference", reference)
        continuum.add_annotation("hypothesis", hypothesis)
        gamma_results = continuum.compute_gamma(self.dissim,
                                                precision_level="medium",
                                                ground_truth_annotators=SortedSet(["reference"]),
                                                soft=True)
        return gamma_results.observed_disorder, gamma_results.observed_disorder


class GammaCategorizationError(BaseGammaMetric, UEMSupportMixin):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dissim = CombinedCategoricalDissimilarity(
            pos_dissim=PositionalSporadicDissimilarity(delta_empty=1.0),
            cat_dissim=AbsoluteCategoricalDissimilarity(delta_empty=1.0)
        )

    @classmethod
    def metric_name(cls) -> str:
        return "GammaCat"

    def _compute_gamma_cat(self, gamma_result: GammaResults) -> Tuple[float, float]:
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as p:

            observed_disorder_job = p.submit(_compute_gamma_k_job,
                                             *(gamma_result.dissimilarity,
                                               gamma_result.best_alignment, None))

            chance_disorders_jobs = [
                p.submit(_compute_gamma_k_job,
                         *(gamma_result.dissimilarity, alignment, None))
                for alignment in gamma_result.chance_alignments
            ]
            observed_disorder = observed_disorder_job.result()
            if observed_disorder == 0:
                return 0, 1
            expected_disorder = float(np.mean(np.array([job_res.result() for job_res in chance_disorders_jobs])))
        if expected_disorder == 0:
            return 0, 0
        return observed_disorder, expected_disorder

    def compute_gamma(self,
                      reference: Annotation,
                      hypothesis: Annotation,
                      ) -> Tuple[float, float]:
        continuum = Continuum()
        continuum.add_annotation("reference", reference)
        continuum.add_annotation("hypothesis", hypothesis)
        gamma_results = continuum.compute_gamma(self.dissim,
                                                precision_level="medium",
                                                ground_truth_annotators=SortedSet(["reference"]),
                                                soft=True)  # TODO: find out if soft or not for this one

        return self._compute_gamma_cat(gamma_results)
