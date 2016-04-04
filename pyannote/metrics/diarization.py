#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2016 CNRS

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

from __future__ import unicode_literals

"""Metrics for diarization"""

import numpy as np
from xarray import DataArray

from pyannote.algorithms.tagging.mapping import HungarianMapper
from pyannote.algorithms.tagging.mapping import GreedyMapper

from .base import BaseMetric
from .identification import IdentificationErrorRate

DER_NAME = 'diarization error rate'


class DiarizationErrorRate(IdentificationErrorRate):
    """Diarization error rate

    First, the optimal mapping between reference and hypothesis labels
    is obtained using the Hungarian algorithm. Then, the actual diarization
    error rate is computed as the identification error rate with each hypothesis
    label translated into the corresponding reference label.

    * Diarization error rate between `reference` and `hypothesis` annotations

        >>> metric = DiarizationErrorRate()
        >>> reference = Annotation(...)           # doctest: +SKIP
        >>> hypothesis = Annotation(...)          # doctest: +SKIP
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Compute global diarization error rate and confidence interval
      over multiple documents

        >>> for reference, hypothesis in ...      # doctest: +SKIP
        ...    metric(reference, hypothesis)      # doctest: +SKIP
        >>> global_value = abs(metric)            # doctest: +SKIP
        >>> mean, (lower, upper) = metric.confidence_interval() # doctest: +SKIP

    * Get diarization error rate detailed components

        >>> components = metric(reference, hypothesis, detailed=True) #doctest +SKIP

    * Get accumulated components

        >>> components = metric[:]                # doctest: +SKIP
        >>> metric['confusion']                   # doctest: +SKIP

    See Also
    --------
    :class:`pyannote.metric.base.BaseMetric`: details on accumulation
    :class:`pyannote.metric.identification.IdentificationErrorRate`: identification error rate

    """

    @classmethod
    def metric_name(cls):
        return DER_NAME

    def __init__(self, **kwargs):
        super(DiarizationErrorRate, self).__init__()
        self._mapper = HungarianMapper()

    def optimal_mapping(self, reference, hypothesis):
        """Optimal label mapping"""
        return self._mapper(hypothesis, reference)

    def _get_details(self, reference, hypothesis, **kwargs):
        reference = reference.anonymize_labels(generator='string')
        hypothesis = hypothesis.anonymize_labels(generator='int')
        mapping = self.optimal_mapping(reference, hypothesis)
        return super(DiarizationErrorRate, self)\
            ._get_details(reference, hypothesis.translate(mapping))


class GreedyDiarizationErrorRate(IdentificationErrorRate):
    """Greedy diarization error rate

    First, the greedy mapping between reference and hypothesis labels is
    obtained. Then, the actual diarization error rate is computed as the
    identification error rate with each hypothesis label translated into the
    corresponding reference label.

    * Greedy diarization error rate between `reference` and `hypothesis` annotations

        >>> metric = GreedyDiarizationErrorRate()
        >>> reference = Annotation(...)           # doctest: +SKIP
        >>> hypothesis = Annotation(...)          # doctest: +SKIP
        >>> value = metric(reference, hypothesis) # doctest: +SKIP

    * Compute global greedy diarization error rate and confidence interval
      over multiple documents

        >>> for reference, hypothesis in ...      # doctest: +SKIP
        ...    metric(reference, hypothesis)      # doctest: +SKIP
        >>> global_value = abs(metric)            # doctest: +SKIP
        >>> mean, (lower, upper) = metric.confidence_interval() # doctest: +SKIP

    * Get greedy diarization error rate detailed components

        >>> components = metric(reference, hypothesis, detailed=True) #doctest +SKIP

    * Get accumulated components

        >>> components = metric[:]                # doctest: +SKIP
        >>> metric['confusion']                   # doctest: +SKIP

    See Also
    --------
    :class:`pyannote.metric.base.BaseMetric`: details on accumulation

    """

    @classmethod
    def metric_name(cls):
        return DER_NAME

    def __init__(self, **kwargs):
        super(GreedyDiarizationErrorRate, self).__init__()
        self._mapper = GreedyMapper()

    def greedy_mapping(self, reference, hypothesis):
        """Greedy label mapping"""
        return self._mapper(hypothesis, reference)

    def _get_details(self, reference, hypothesis, **kwargs):
        reference = reference.anonymize_labels(generator='string')
        hypothesis = hypothesis.anonymize_labels(generator='int')
        mapping = self.greedy_mapping(reference, hypothesis)
        return super(GreedyDiarizationErrorRate, self)\
            ._get_details(reference, hypothesis.translate(mapping))


PURITY_NAME = 'purity'
PURITY_TOTAL = 'total'
PURITY_CORRECT = 'correct'


class DiarizationPurity(BaseMetric):
    """Purity

    Compute purity of hypothesis clusters with respect to reference classes.

    Parameters
    ----------
    per_cluster : bool, optional
        By default (per_cluster = False), clusters are duration-weighted.
        When per_cluster = True, each cluster is given the same weight.

    """

    @classmethod
    def metric_name(cls):
        return PURITY_NAME

    @classmethod
    def metric_components(cls):
        return [PURITY_TOTAL, PURITY_CORRECT]

    def __init__(self, detection_error=False, per_cluster=False, **kwargs):
        super(DiarizationPurity, self).__init__()
        self.per_cluster = per_cluster

    def _get_details(self, reference, hypothesis, **kwargs):
        detail = self._init_details()

        matrix = reference.smooth() * hypothesis.smooth()

        biggest = matrix.max(dim='i')
        duration = DataArray(
            [hypothesis.label_duration(j.item()) for j in biggest.coords['j']],
            coords=biggest.coords)

        if self.per_cluster:
            # biggest class in each cluster
            detail[PURITY_CORRECT] = (biggest / duration).sum().item()
            # number of clusters (as float)
            detail[PURITY_TOTAL] = len(biggest)

        else:
            if np.prod(matrix.shape):
                detail[PURITY_CORRECT] = biggest.sum().item()
            else:
                detail[PURITY_CORRECT] = 0.
            # total duration of clusters (with overlap)
            detail[PURITY_TOTAL] = duration.sum().item()

        return detail

    def _get_rate(self, detail):
        if detail[PURITY_TOTAL] > 0.:
            return detail[PURITY_CORRECT] / detail[PURITY_TOTAL]
        else:
            return 1.

    def _pretty(self, detail):
        string = ""
        if self.per_cluster:
            string += "  - clusters: %d\n" % (detail[PURITY_TOTAL])
            string += "  - correct: %.2f\n" % (detail[PURITY_CORRECT])
        else:
            string += "  - duration: %.2f seconds\n" % (detail[PURITY_TOTAL])
            string += "  - correct: %.2f seconds\n" % (detail[PURITY_CORRECT])
        string += "  - %s: %.2f %%\n" % (self.name, 100*detail[self.name])
        return string

COVERAGE_NAME = 'coverage'


class DiarizationCoverage(DiarizationPurity):
    """Coverage

    Compute coverage of hypothesis clusters with respect to reference classes
    (i.e. purity of reference classes with respect to hypothesis clusters)

    Parameters
    ----------
    per_cluster : bool, optional
        By default (per_cluster = False), classes are duration-weighted.
        When per_cluster = True, each class is given the same weight.
    """

    @classmethod
    def metric_name(cls):
        return COVERAGE_NAME

    def __init__(self, per_cluster=False, **kwargs):
        super(DiarizationCoverage, self).__init__(per_cluster=per_cluster)

    def _get_details(self, reference, hypothesis, **kwargs):
        return super(DiarizationCoverage, self)\
            ._get_details(hypothesis, reference)

    def _pretty(self, detail):
        string = ""
        if self.per_cluster:
            string += "  - classes: %d\n" % (detail[PURITY_TOTAL])
            string += "  - correct: %.2f\n" % (detail[PURITY_CORRECT])
        else:
            string += "  - duration: %.2f seconds\n" % (detail[PURITY_TOTAL])
            string += "  - correct: %.2f seconds\n" % (detail[PURITY_CORRECT])
        string += "  - %s: %.2f %%\n" % (self.name, 100*detail[self.name])
        return string


HOMOGENEITY_NAME = 'homogeneity'
HOMOGENEITY_ENTROPY = 'entropy'
HOMOGENEITY_CROSS_ENTROPY = 'cross-entropy'


class DiarizationHomogeneity(BaseMetric):
    """Homogeneity"""

    @classmethod
    def metric_name(cls):
        return HOMOGENEITY_NAME

    @classmethod
    def metric_components(cls):
        return [HOMOGENEITY_ENTROPY, HOMOGENEITY_CROSS_ENTROPY]

    def _get_details(self, reference, hypothesis, **kwargs):
        detail = self._init_details()

        matrix = reference.smooth() * hypothesis.smooth()

        duration = matrix.sum()
        rduration = matrix.sum(dim='j')
        hduration = matrix.sum(dim='i')

        # Reference entropy and reference/hypothesis cross-entropy
        ratio = rduration / duration
        entropy = -(ratio * np.log(ratio)).sum()
        detail[HOMOGENEITY_ENTROPY] = entropy.item()

        cross_entropy = -((matrix / duration) * np.log(matrix / hduration)).sum()
        detail[HOMOGENEITY_CROSS_ENTROPY] = cross_entropy.item()

        return detail

    def _get_rate(self, detail):
        numerator = 1. * detail[HOMOGENEITY_CROSS_ENTROPY]
        denominator = 1. * detail[HOMOGENEITY_ENTROPY]
        if denominator == 0.:
            if numerator == 0:
                return 1.
            else:
                return 0.
        else:
            return 1. - numerator/denominator

    def _pretty(self, detail):
        string = ""
        string += "  - %s: %.2f\n" % \
                  (HOMOGENEITY_ENTROPY, detail[HOMOGENEITY_ENTROPY])
        string += "  - %s: %.2f\n" % \
                  (HOMOGENEITY_CROSS_ENTROPY, detail[HOMOGENEITY_CROSS_ENTROPY])
        string += "  - %s: %.2f %%\n" % (self.name, 100*detail[self.name])
        return string

COMPLETENESS_NAME = 'completeness'


class DiarizationCompleteness(DiarizationHomogeneity):
    """Completeness"""

    @classmethod
    def metric_name(cls):
        return COMPLETENESS_NAME

    def _get_details(self, reference, hypothesis, **kwargs):
        return super(DiarizationCompleteness, self)\
            ._get_details(hypothesis, reference)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
