from pyannote.metrics.base import BaseMetric
from pyannote.metrics.types import Details, MetricComponents
from pyannote.metrics.utils import UEMSupportMixin

GAMMA_COMPONENTS = [
    "DISORDER", "CHANCE_DISORDER"
]

class GammaMetricMixin:

    pass


class GammaDetectionErrorRate(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls) -> str:
        return "GammaDER"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return GAMMA_COMPONENTS

    def compute_components(self, reference, hypothesis, **kwargs) -> Details:
        pass

    def compute_metric(self, components: Details):
        pass


class GammaIdentificationErrorRate(UEMSupportMixin, BaseMetric):
    @classmethod
    def metric_name(cls) -> str:
        return "GammaIER"

    @classmethod
    def metric_components(cls) -> MetricComponents:
        return GAMMA_COMPONENTS

    def compute_components(self, reference, hypothesis, **kwargs) -> Details:
        pass

    def compute_metric(self, components: Details):
        pass

