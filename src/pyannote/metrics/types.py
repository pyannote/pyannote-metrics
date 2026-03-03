from typing import Dict, List

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

MetricComponent = str
CalibrationMethod = Literal["isotonic", "sigmoid"]
MetricComponents = List[MetricComponent]
Details = Dict[MetricComponent, float]