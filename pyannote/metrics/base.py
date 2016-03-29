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

import scipy.stats
import numpy as np
from pyannote.algorithms.tagging import DirectTagger


class BaseMetric(object):
    """
    :class:`BaseMetric` is the base class for most PyAnnote evaluation metrics.

    Parameters
    ----------
    name : str
        Human-readable name of the metric (eg. 'diarization error rate')
    components : list, set or tuple
        Human-readable names of the components of the metric
        (eg. ['correct', 'false alarm', 'miss', 'confusion'])

    """

    @classmethod
    def metric_name(cls):
        raise NotImplementedError("Missing class method 'metric_name'.")

    @classmethod
    def metric_components(cls):
        raise NotImplementedError("Missing class method 'metric_components'")

    def __init__(self, **kwargs):
        super(BaseMetric, self).__init__()
        self.__name = self.__class__.metric_name()
        self.__values = set(self.__class__.metric_components())
        self.reset()
        self._tagger = DirectTagger()

    def __get_name(self):
        return self.__class__.metric_name()
    name = property(fget=__get_name, doc="Metric name.")

    def __accumulate(self, components):
        """Accumulate metric components

        Parameters
        ----------
        components : dict
            Dictionary where keys are component names and values are component
            values

        """
        for name in self.__values:
            self.__details[name] += components[name]

    def __compute(self, components, accumulate=True, detailed=False):
        """Compute metric value from computed `components`

        Parameters
        ----------
        components : dict
            Dictionary where keys are components names and values are component
            values
        accumulate : bool, optional
            If True, components are accumulated. Defaults to True.
        detailed : bool, optional
            By default (False), return metric value only.
            Set `detailed` to True to return `components` updated with metric
            value (for key `self.name`).

        Returns
        -------
        value (if `detailed` is False) : float
            Metric value
        components (if `detailed` is True) : dict
            `components` updated with metric value
        """
        if accumulate:
            self.__accumulate(components)
        rate = self._get_rate(components)
        if detailed:
            components = dict(components)
            components.update({self.__name: rate})
            return components
        else:
            return rate

    def __call__(self, reference, hypothesis, detailed=False, **kwargs):
        """Compute metric value and accumulate components

        Parameters
        ----------
        reference : type depends on the metric
            Manual `reference`
        hypothesis : same as `reference`
            Evaluated `hypothesis`
        detailed : bool, optional
            By default (False), return metric value only.

            Set `detailed` to True to return dictionary where keys are
            components names and values are component values


        Returns
        -------
        value : float (if `detailed` is False)
            Metric value
        components : dict (if `detailed` is True)
            `components` updated with metric value

        """
        detail = self._get_details(reference, hypothesis, **kwargs)
        self.__rates.append((reference.uri, self._get_rate(detail)))
        return self.__compute(detail, accumulate=True, detailed=detailed)

    def __str__(self):
        detail = self.__compute(self.__details,
                                accumulate=False,
                                detailed=True)
        return self._pretty(detail)

    def __abs__(self):
        """Compute metric value from accumulated components"""
        return self._get_rate(self.__details)

    def __getitem__(self, component):
        """Get value of accumulated `component`.

        Parameters
        ----------
        component : str
            Name of `component`

        Returns
        -------
        value : type depends on the metric
            Value of accumulated `component`

        """
        if component == slice(None, None, None):
            return dict(self.__details)
        else:
            return self.__details[component]

    def __iter__(self):
        """Iterator over the accumulated (uri, value)"""
        for v, r in self.__rates:
            yield v, r

    def _get_details(self, reference, hypothesis, **kwargs):
        """Compute metric components

        Parameters
        ----------
        reference : type depends on the metric
            Manual `reference`
        hypothesis : same as `reference`
            Evaluated `hypothesis`

        Returns
        -------
        components : dict
            Dictionary where keys are component names and values are component
            values

        """
        raise NotImplementedError("Missing method '_get_details'.")

    def _get_rate(self, components):
        """Compute metric value from computed `components`

        Parameters
        ----------
        components : dict
            Dictionary where keys are components names and values are component
            values

        Returns
        -------
        value : type depends on the metric
            Metric value
        """
        raise NotImplementedError("Missing method '_get_rate'.")

    def _pretty(self, components):
        string = '%s: %g\n' % (self.name, components[self.name])
        for name, value in components.items():
            if name == self.name:
                continue
            string += '  - %s: %g\n' % (name, value)
        return string

    def _init_details(self):
        return {value: 0. for value in self.__values}

    def reset(self):
        """Reset accumulated components and metric values"""
        self.__details = self._init_details()
        self.__rates = []

    def confidence_interval(self, alpha=0.9):
        """Compute confidence interval on accumulated metric values

        Parameters
        ----------
        alpha : float, optional
            Probability that the returned confidence interval contains
            the true metric value.

        Returns
        -------
        (center, (lower, upper))
            with center the mean of the conditional pdf of the metric value
            and (lower, upper) is a confidence interval centered on the median,
            containing the estimate to a probability alpha.

        See Also:
        ---------
        scipy.stats.bayes_mvs

        """
        m, _, _ = scipy.stats.bayes_mvs([r for _, r in self.__rates], alpha=alpha)
        return m


PRECISION_NAME = 'precision'
PRECISION_RETRIEVED = '# retrieved'
PRECISION_RELEVANT_RETRIEVED = '# relevant retrieved'


class Precision(BaseMetric):
    """
    :class:`Precision` is a base class for precision-like evaluation metrics.

    It defines two components '# retrieved' and '# relevant retrieved' and the
    _get_rate() method to compute the actual precision:

        Precision = # retrieved / # relevant retrieved

    Inheriting classes must implement _get_details().
    """

    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls):
        return [PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED]

    def _get_rate(self, components):
        """Compute precision from `components`"""
        numerator = components[PRECISION_RELEVANT_RETRIEVED]
        denominator = components[PRECISION_RETRIEVED]
        if denominator == 0.:
            if numerator == 0:
                return 1.
            else:
                raise ValueError('')
        else:
            return numerator/denominator

RECALL_NAME = 'recall'
RECALL_RELEVANT = '# relevant'
RECALL_RELEVANT_RETRIEVED = '# relevant retrieved'


class Recall(BaseMetric):
    """
    :class:`Recall` is a base class for recall-like evaluation metrics.

    It defines two components '# relevant' and '# relevant retrieved' and the
    _get_rate() method to compute the actual recall:

        Recall = # relevant retrieved / # relevant

    Inheriting classes must implement _get_details().
    """

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    @classmethod
    def metric_components(cls):
        return [RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED]

    def _get_rate(self, components):
        """Compute recall from `components`"""
        numerator = components[RECALL_RELEVANT_RETRIEVED]
        denominator = components[RECALL_RELEVANT]
        if denominator == 0.:
            if numerator == 0:
                return 1.
            else:
                raise ValueError('')
        else:
            return numerator/denominator


def f_measure(precision, recall, beta=1.):
    """Compute f-measure

    f-measure is defined as follows:
        F(P, R, b) = (1+b²).P.R / (b².P + R)

    where P is `precision`, R is `recall` and b is `beta`
    """
    return (1+beta*beta)*precision*recall / (beta*beta*precision+recall)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
