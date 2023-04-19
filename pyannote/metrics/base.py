#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)

# Copyright (c) 2012- CNRS

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
import inspect
from typing import List, Union, Optional, Set, Tuple

import warnings
import numpy as np
import pandas as pd
import scipy.stats
from pyannote.core import Annotation, Timeline

from pyannote.metrics.types import Details, MetricComponents


def clone(metric):
    """Construct a new empty metric with the same parameters as `metric`.

    Clone does a deep copy of the metric without actually copying any
    results or accumulators. It returns a new metric (with the same
    parameters) that has not yet been used to evaluate any data.

    Parameters
    ----------
    metric : metric instance
        The metric to be cloned.

    Returns
    -------
    metric : object
        The deep copy of the metric,
    """
    if hasattr(metric, '__pyannote_clone__') and not inspect.isclass(metric):
        # If metric implements a custom cloning method, default to that.
        return metric.__pyannote_clone()
    cls = metric.__class__
    new_metric_params = metric.get_params()
    new_metric = cls(**new_metric_params)

    # Sanity check that all parameters were saved by the constructor without
    # modification.
    actual_params = new_metric.get_params()
    for pname in new_metric_params:
        error_msg = (
            f'Cannot clone metric {repr(metric)}, as the constructor either '
            f'does not set or modifies parameter {pname}.')
        print(error_msg)
        expected_param = new_metric_params[pname]
        try:
            actual_param = actual_params[pname]
        except KeyError:
            raise RuntimeError(error_msg())
        if expected_param != actual_param:
            raise RuntimeError(error_msg())

    return new_metric


class BaseMetric:
    """
    :class:`BaseMetric` is the base class for most pyannote evaluation metrics.

    Attributes
    ----------
    name : str
        Human-readable name of the metric (eg. 'diarization error rate')
    """

    @classmethod
    def metric_name(cls) -> str:
        raise NotImplementedError(
            cls.__name__ + " is missing a 'metric_name' class method. "
                           "It should return the name of the metric as string."
        )

    @classmethod
    def metric_components(cls) -> MetricComponents:
        raise NotImplementedError(
            cls.__name__ + " is missing a 'metric_components' class method. "
                           "It should return the list of names of metric components."
        )

    def __init__(self, **kwargs):
        super(BaseMetric, self).__init__()
        self.metric_name_ = self.__class__.metric_name()
        self.components_: Set[str] = set(self.__class__.metric_components())
        self.reset()

    @classmethod
    def _get_param_names(cls):
        pnames = set()
        for cls_ in cls.__mro__:
            init = cls_.__init__
            init_signature = inspect.signature(init)
            for p in init_signature.parameters.values():
                if p.name == 'self':
                    continue
                if p.kind in {p.VAR_POSITIONAL, p.VAR_KEYWORD}:
                    # Skip *args/**kwargs.
                    continue
                pnames.add(p.name)
        return pnames

    def get_params(self):
        """Return parameters for this metric.

        Returns
        -------
        params : dict
            Mapping from parameter names to values.
        """
        return {pname: getattr(self, pname)
                for pname in self._get_param_names()}


    def init_components(self):
        return {value: 0.0 for value in self.components_}

    def reset(self):
        """Reset accumulated components and metric values"""
        self.accumulated_: Details = dict()
        self.results_: List = list()
        for value in self.components_:
            self.accumulated_[value] = 0.0

    @property
    def name(self):
        """Metric name."""
        return self.metric_name()

    def __call__(self, reference: Union[Timeline, Annotation],
                 hypothesis: Union[Timeline, Annotation],
                 detailed: bool = False, uri: Optional[str] = None, **kwargs):
        """Compute metric value and accumulate components

        Parameters
        ----------
        reference : type depends on the metric
            Manual `reference`
        hypothesis : type depends on the metric
            Evaluated `hypothesis`
        uri : optional
            Override uri.
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

        # compute metric components
        components = self.compute_components(reference, hypothesis, **kwargs)

        # compute rate based on components
        components[self.metric_name_] = self.compute_metric(components)

        # keep track of this computation
        uri = uri or getattr(reference, "uri", "NA")
        self.results_.append((uri, components))

        # accumulate components
        for name in self.components_:
            self.accumulated_[name] += components[name]

        if detailed:
            return components

        return components[self.metric_name_]

    def report(self, display: bool = False) -> pd.DataFrame:
        """Evaluation report

        Parameters
        ----------
        display : bool, optional
            Set to True to print the report to stdout.

        Returns
        -------
        report : pandas.DataFrame
            Dataframe with one column per metric component, one row per
            evaluated item, and one final row for accumulated results.
        """

        report = []
        uris = []

        percent = "total" in self.metric_components()

        for uri, components in self.results_:
            row = {}
            if percent:
                total = components["total"]
            for key, value in components.items():
                if key == self.name:
                    row[key, "%"] = 100 * value
                elif key == "total":
                    row[key, ""] = value
                else:
                    row[key, ""] = value
                    if percent:
                        if total > 0:
                            row[key, "%"] = 100 * value / total
                        else:
                            row[key, "%"] = np.NaN

            report.append(row)
            uris.append(uri)

        row = {}
        components = self.accumulated_

        if percent:
            total = components["total"]

        for key, value in components.items():
            if key == self.name:
                row[key, "%"] = 100 * value
            elif key == "total":
                row[key, ""] = value
            else:
                row[key, ""] = value
                if percent:
                    if total > 0:
                        row[key, "%"] = 100 * value / total
                    else:
                        row[key, "%"] = np.NaN

        row[self.name, "%"] = 100 * abs(self)
        report.append(row)
        uris.append("TOTAL")

        df = pd.DataFrame(report)

        df["item"] = uris
        df = df.set_index("item")

        df.columns = pd.MultiIndex.from_tuples(df.columns)

        df = df[[self.name] + self.metric_components()]

        if display:
            print(
                df.to_string(
                    index=True,
                    sparsify=False,
                    justify="right",
                    float_format=lambda f: "{0:.2f}".format(f),
                )
            )

        return df

    def __str__(self):
        report = self.report(display=False)
        return report.to_string(
            sparsify=False, float_format=lambda f: "{0:.2f}".format(f)
        )

    def __repr__(self):
        cls = self.__class__.__name__
        params = self.get_params()
        pnames = sorted(params)
        signature = ', '.join(f'{pname}={params[pname]}' for pname in pnames)
        return f'{cls}({signature})'

    def __abs__(self):
        """Compute metric value from accumulated components"""
        return self.compute_metric(self.accumulated_)

    def __getitem__(self, component: str) -> Union[float, Details]:
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
            return dict(self.accumulated_)
        else:
            return self.accumulated_[component]

    def __iter__(self):
        """Iterator over the accumulated (uri, value)"""
        for uri, component in self.results_:
            yield uri, component

    def __add__(self, other):
        cls = self.__class__
        result = cls()
        result.results_ = self.results_ + other.results_
        for cname in self.components_:
            result.accumulated_[cname] += self.accumulated_[cname]
            result.accumulated_[cname] += other.accumulated_[cname]
        return result

    def __radd__(self, other):
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def compute_components(self,
                           reference: Union[Timeline, Annotation],
                           hypothesis: Union[Timeline, Annotation],
                           **kwargs) -> Details:
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
        raise NotImplementedError(
            self.__class__.__name__ + " is missing a 'compute_components' method."
                                      "It should return a dictionary where keys are component names "
                                      "and values are component values."
        )

    def compute_metric(self, components: Details):
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
        raise NotImplementedError(
            self.__class__.__name__ + " is missing a 'compute_metric' method. "
                                      "It should return the actual value of the metric based "
                                      "on the precomputed component dictionary given as input."
        )

    def confidence_interval(self, alpha: float = 0.9) \
            -> Tuple[float, Tuple[float, float]]:
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

        values = [r[self.metric_name_] for _, r in self.results_]

        if len(values) == 0:
            raise ValueError("Please evaluate a bunch of files before computing confidence interval.")

        elif len(values) == 1:
            warnings.warn("Cannot compute a reliable confidence interval out of just one file.")
            center = lower = upper = values[0]
            return center, (lower, upper)

        else:
            return scipy.stats.bayes_mvs(values, alpha=alpha)[0]


PRECISION_NAME = "precision"
PRECISION_RETRIEVED = "# retrieved"
PRECISION_RELEVANT_RETRIEVED = "# relevant retrieved"


class Precision(BaseMetric):
    """
    :class:`Precision` is a base class for precision-like evaluation metrics.

    It defines two components '# retrieved' and '# relevant retrieved' and the
    compute_metric() method to compute the actual precision:

        Precision = # retrieved / # relevant retrieved

    Inheriting classes must implement compute_components().
    """

    @classmethod
    def metric_name(cls):
        return PRECISION_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [PRECISION_RETRIEVED, PRECISION_RELEVANT_RETRIEVED]

    def compute_metric(self, components: Details) -> float:
        """Compute precision from `components`"""
        numerator = components[PRECISION_RELEVANT_RETRIEVED]
        denominator = components[PRECISION_RETRIEVED]
        if denominator == 0.0:
            if numerator == 0:
                return 1.0
            else:
                raise ValueError("")
        else:
            return numerator / denominator


RECALL_NAME = "recall"
RECALL_RELEVANT = "# relevant"
RECALL_RELEVANT_RETRIEVED = "# relevant retrieved"


class Recall(BaseMetric):
    """
    :class:`Recall` is a base class for recall-like evaluation metrics.

    It defines two components '# relevant' and '# relevant retrieved' and the
    compute_metric() method to compute the actual recall:

        Recall = # relevant retrieved / # relevant

    Inheriting classes must implement compute_components().
    """

    @classmethod
    def metric_name(cls):
        return RECALL_NAME

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return [RECALL_RELEVANT, RECALL_RELEVANT_RETRIEVED]

    def compute_metric(self, components: Details) -> float:
        """Compute recall from `components`"""
        numerator = components[RECALL_RELEVANT_RETRIEVED]
        denominator = components[RECALL_RELEVANT]
        if denominator == 0.0:
            if numerator == 0:
                return 1.0
            else:
                raise ValueError("")
        else:
            return numerator / denominator


def f_measure(precision: float, recall: float, beta=1.0) -> float:
    """Compute f-measure

    f-measure is defined as follows:
        F(P, R, b) = (1+b²).P.R / (b².P + R)

    where P is `precision`, R is `recall` and b is `beta`
    """
    if precision + recall == 0.0:
        return 0
    return (1 + beta * beta) * precision * recall / (beta * beta * precision + recall)
