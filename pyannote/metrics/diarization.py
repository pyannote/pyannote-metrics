#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012-2017 CNRS

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

from .matcher import HungarianMapper
from .matcher import GreedyMapper

from .base import BaseMetric, f_measure
from .utils import UEMSupportMixin
from .identification import IdentificationErrorRate

DER_NAME = 'diarization error rate'


class DiarizationErrorRate(IdentificationErrorRate):
    """Diarization error rate

    First, the optimal mapping between reference and hypothesis labels
    is obtained using the Hungarian algorithm. Then, the actual diarization
    error rate is computed as the identification error rate with each hypothesis
    label translated into the corresponding reference label.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    Usage
    -----

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

    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super(DiarizationErrorRate, self).__init__(
            collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = HungarianMapper()

    def optimal_mapping(self, reference, hypothesis, uem=None):
        """Optimal label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map

        Returns
        -------
        mapping : dict
            Mapping between hypothesis (key) and reference (value) labels
        """

        # NOTE that this 'uemification' will not be called when
        # 'optimal_mapping' is called from 'compute_components' as it
        # has already been done in 'compute_components'
        if uem:
            reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)

        # call hungarian mapper
        mapping = self.mapper_(hypothesis, reference)
        return mapping

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):

        # crop reference and hypothesis to evaluated regions (uem)
        # remove collars around reference segment boundaries
        # remove overlap regions (if requested)
        reference, hypothesis, uem = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap,
            returns_uem=True)
        # NOTE that this 'uemification' must be done here because it
        # might have an impact on the search for the optimal mapping.

        # make sure reference only contains string labels ('A', 'B', ...)
        reference = reference.anonymize_labels(generator='string')

        # make sure hypothesis only contains integer labels (1, 2, ...)
        hypothesis = hypothesis.anonymize_labels(generator='int')

        # optimal (int --> str) mapping
        mapping = self.optimal_mapping(reference, hypothesis)

        # compute identification error rate based on mapped hypothesis
        # NOTE that collar is set to 0.0 because 'uemify' has already
        # been applied (same reason for setting skip_overlap to False)
        mapped = hypothesis.translate(mapping)
        return super(DiarizationErrorRate, self)\
            .compute_components(reference, mapped, uem=uem,
                                collar=0.0, skip_overlap=False,
                                **kwargs)


class GreedyDiarizationErrorRate(IdentificationErrorRate):
    """Greedy diarization error rate

    First, the greedy mapping between reference and hypothesis labels is
    obtained. Then, the actual diarization error rate is computed as the
    identification error rate with each hypothesis label translated into the
    corresponding reference label.

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    Usage
    -----

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

    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super(GreedyDiarizationErrorRate, self).__init__(
            collar=collar, skip_overlap=skip_overlap, **kwargs)
        self.mapper_ = GreedyMapper()

    def greedy_mapping(self, reference, hypothesis, uem=None):
        """Greedy label mapping

        Parameters
        ----------
        reference : Annotation
        hypothesis : Annotation
            Reference and hypothesis diarization
        uem : Timeline
            Evaluation map

        Returns
        -------
        mapping : dict
            Mapping between hypothesis (key) and reference (value) labels
        """
        if uem:
            reference, hypothesis = self.uemify(reference, hypothesis, uem=uem)
        return self.mapper_(hypothesis, reference)

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):

        # crop reference and hypothesis to evaluated regions (uem)
        # remove collars around reference segment boundaries
        # remove overlap regions (if requested)
        reference, hypothesis, uem = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap,
            returns_uem=True)
        # NOTE that this 'uemification' must be done here because it
        # might have an impact on the search for the greedy mapping.

        # make sure reference only contains string labels ('A', 'B', ...)
        reference = reference.anonymize_labels(generator='string')

        # make sure hypothesis only contains integer labels (1, 2, ...)
        hypothesis = hypothesis.anonymize_labels(generator='int')

        # greedy (int --> str) mapping
        mapping = self.greedy_mapping(reference, hypothesis)

        # compute identification error rate based on mapped hypothesis
        # NOTE that collar is set to 0.0 because 'uemify' has already
        # been applied (same reason for setting skip_overlap to False)
        mapped = hypothesis.translate(mapping)
        return super(GreedyDiarizationErrorRate, self)\
            .compute_components(reference, mapped, uem=uem,
                                collar=0.0, skip_overlap=False,
                                **kwargs)

PURITY_NAME = 'purity'
PURITY_TOTAL = 'total'
PURITY_CORRECT = 'correct'


class DiarizationPurity(UEMSupportMixin, BaseMetric):
    """Cluster purity

    A hypothesized annotation has perfect purity if all of its labels overlap
    only segments which are members of a single reference label.

    Parameters
    ----------
    weighted : bool, optional
        When True (default), each cluster is weighted by its overall duration.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return PURITY_NAME

    @classmethod
    def metric_components(cls):
        return [PURITY_TOTAL, PURITY_CORRECT]

    def __init__(self, collar=0.0, skip_overlap=False,
                 weighted=True, **kwargs):
        super(DiarizationPurity, self).__init__(**kwargs)
        self.weighted = weighted
        self.collar = collar
        self.skip_overlap = skip_overlap

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):

        detail = self.init_components()

        # crop reference and hypothesis to evaluated regions (uem)
        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap)

        if not reference:
            return detail

        # cooccurrence matrix
        matrix = reference * hypothesis

        # duration of largest class in each cluster
        largest = matrix.max(dim='i')
        duration = matrix.sum(dim='i')

        if self.weighted:
            detail[PURITY_CORRECT] = 0.
            if np.prod(matrix.shape):
                detail[PURITY_CORRECT] = largest.sum().item()
            detail[PURITY_TOTAL] = duration.sum().item()

        else:
            detail[PURITY_CORRECT] = (largest / duration).sum().item()
            detail[PURITY_TOTAL] = len(largest)

        return detail

    def compute_metric(self, detail):
        if detail[PURITY_TOTAL] > 0.:
            return detail[PURITY_CORRECT] / detail[PURITY_TOTAL]
        return 1.


COVERAGE_NAME = 'coverage'


class DiarizationCoverage(DiarizationPurity):
    """Cluster coverage

    A hypothesized annotation has perfect coverage if all segments from a
    given reference label are clustered in the same cluster.

    Parameters
    ----------
    weighted : bool, optional
        When True (default), each cluster is weighted by its overall duration.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    """

    @classmethod
    def metric_name(cls):
        return COVERAGE_NAME

    def __init__(self, collar=0.0, skip_overlap=False,
                 weighted=True, **kwargs):
        super(DiarizationCoverage, self).__init__(
            collar=collar, skip_overlap=skip_overlap,
            weighted=weighted, **kwargs)

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):
        return super(DiarizationCoverage, self)\
            .compute_components(hypothesis, reference, uem=uem, **kwargs)


PURITY_COVERAGE_NAME = 'F[purity|coverage]'
PURITY_COVERAGE_LARGEST_CLASS = 'largest_class'
PURITY_COVERAGE_TOTAL_CLUSTER = 'total_cluster'
PURITY_COVERAGE_LARGEST_CLUSTER = 'largest_cluster'
PURITY_COVERAGE_TOTAL_CLASS = 'total_class'


class DiarizationPurityCoverageFMeasure(UEMSupportMixin, BaseMetric):
    """Compute diarization purity and coverage, and return their F-score.

    Parameters
    ----------
    weighted : bool, optional
        When True (default), each cluster/class is weighted by its overall
        duration.
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).
    beta : float, optional
        When beta > 1, greater importance is given to coverage.
        When beta < 1, greater importance is given to purity.
        Defaults to 1.

    See also
    --------
    pyannote.metrics.diarization.DiarizationPurity
    pyannote.metrics.diarization.DiarizationCoverage
    pyannote.metrics.base.f_measure

    """

    @classmethod
    def metric_name(cls):
        return PURITY_COVERAGE_NAME

    @classmethod
    def metric_components(cls):
        return [PURITY_COVERAGE_LARGEST_CLASS,
                PURITY_COVERAGE_TOTAL_CLUSTER,
                PURITY_COVERAGE_LARGEST_CLUSTER,
                PURITY_COVERAGE_TOTAL_CLASS]

    def __init__(self, collar=0.0, skip_overlap=False,
                 weighted=True, beta=1., **kwargs):
        super(DiarizationPurityCoverageFMeasure, self).__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap
        self.weighted = weighted
        self.beta = beta

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):

        detail = self.init_components()

        # crop reference and hypothesis to evaluated regions (uem)
        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap)

        # cooccurrence matrix
        matrix = reference * hypothesis

        # duration of largest class in each cluster
        largest_class = matrix.max(dim='i')
        # duration of clusters
        duration_cluster = matrix.sum(dim='i')

        # duration of largest cluster in each class
        largest_cluster = matrix.max(dim='j')
        # duration of classes
        duration_class = matrix.sum(dim='j')

        if self.weighted:
            # compute purity components
            detail[PURITY_COVERAGE_LARGEST_CLASS] = 0.
            if np.prod(matrix.shape):
                detail[PURITY_COVERAGE_LARGEST_CLASS] = largest_class.sum().item()
            detail[PURITY_COVERAGE_TOTAL_CLUSTER] = duration_cluster.sum().item()
            # compute coverage components
            detail[PURITY_COVERAGE_LARGEST_CLUSTER] = 0.
            if np.prod(matrix.shape):
                detail[PURITY_COVERAGE_LARGEST_CLUSTER] = largest_cluster.sum().item()
            detail[PURITY_COVERAGE_TOTAL_CLASS] = duration_class.sum().item()

        else:
            # compute purity components
            detail[PURITY_COVERAGE_LARGEST_CLASS] = (largest_class / duration_cluster).sum().item()
            detail[PURITY_COVERAGE_TOTAL_CLUSTER] = len(largest_class)
            # compute coverage components
            detail[PURITY_COVERAGE_LARGEST_CLUSTER] = (largest_cluster / duration_class).sum().item()
            detail[PURITY_COVERAGE_TOTAL_CLASS] = len(largest_cluster)

        # compute purity
        detail[PURITY_NAME] = \
            1. if detail[PURITY_COVERAGE_TOTAL_CLUSTER] == 0. \
            else detail[PURITY_COVERAGE_LARGEST_CLASS] / detail[PURITY_COVERAGE_TOTAL_CLUSTER]
        # compute coverage
        detail[COVERAGE_NAME] = \
            1. if detail[PURITY_COVERAGE_TOTAL_CLASS] == 0. \
            else detail[PURITY_COVERAGE_LARGEST_CLUSTER] / detail[PURITY_COVERAGE_TOTAL_CLASS]

        return detail

    def compute_metric(self, detail):
        _, _, value = self.compute_metrics(detail=detail)
        return value

    def compute_metrics(self, detail=None):

        detail = self.accumulated_ if detail is None else detail

        purity = \
            1. if detail[PURITY_COVERAGE_TOTAL_CLUSTER] == 0. \
            else detail[PURITY_COVERAGE_LARGEST_CLASS] / detail[PURITY_COVERAGE_TOTAL_CLUSTER]

        coverage = \
            1. if detail[PURITY_COVERAGE_TOTAL_CLASS] == 0. \
            else detail[PURITY_COVERAGE_LARGEST_CLUSTER] / detail[PURITY_COVERAGE_TOTAL_CLASS]

        return purity, coverage, f_measure(purity, coverage, beta=self.beta)


HOMOGENEITY_NAME = 'homogeneity'
HOMOGENEITY_ENTROPY = 'entropy'
HOMOGENEITY_CROSS_ENTROPY = 'cross-entropy'


class DiarizationHomogeneity(UEMSupportMixin, BaseMetric):
    """Cluster homogeneity

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    """

    @classmethod
    def metric_name(cls):
        return HOMOGENEITY_NAME

    @classmethod
    def metric_components(cls):
        return [HOMOGENEITY_ENTROPY, HOMOGENEITY_CROSS_ENTROPY]

    def __init__(self, collar=0.0, skip_overlap=False, **kwargs):
        super(DiarizationHomogeneity, self).__init__(**kwargs)
        self.collar = collar
        self.skip_overlap = skip_overlap

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):

        detail = self.init_components()

        # crop reference and hypothesis to evaluated regions (uem)
        reference, hypothesis = self.uemify(
            reference, hypothesis, uem=uem,
            collar=self.collar, skip_overlap=self.skip_overlap)

        # cooccurrence matrix
        matrix = np.array(reference * hypothesis)

        duration = np.sum(matrix)
        rduration = np.sum(matrix, axis=1)
        hduration = np.sum(matrix, axis=0)

        # reference entropy and reference/hypothesis cross-entropy
        ratio = np.ma.divide(rduration, duration).filled(0.)
        detail[HOMOGENEITY_ENTROPY] = \
            -np.sum(ratio * np.ma.log(ratio).filled(0.))

        ratio = np.ma.divide(matrix, duration).filled(0.)
        hratio = np.ma.divide(matrix, hduration).filled(0.)
        detail[HOMOGENEITY_CROSS_ENTROPY] = \
            -np.sum(ratio * np.ma.log(hratio).filled(0.))

        return detail

    def compute_metric(self, detail):
        numerator = 1. * detail[HOMOGENEITY_CROSS_ENTROPY]
        denominator = 1. * detail[HOMOGENEITY_ENTROPY]
        if denominator == 0.:
            if numerator == 0:
                return 1.
            else:
                return 0.
        else:
            return 1. - numerator / denominator


COMPLETENESS_NAME = 'completeness'


class DiarizationCompleteness(DiarizationHomogeneity):
    """Cluster completeness

    Parameters
    ----------
    collar : float, optional
        Duration (in seconds) of collars removed from evaluation around
        boundaries of reference segments.
    skip_overlap : bool, optional
        Set to True to not evaluate overlap regions.
        Defaults to False (i.e. keep overlap regions).

    """

    @classmethod
    def metric_name(cls):
        return COMPLETENESS_NAME

    def compute_components(self, reference, hypothesis, uem=None, **kwargs):
        return super(DiarizationCompleteness, self)\
            .compute_components(hypothesis, reference, uem=uem, **kwargs)
