from typing import Dict, List

from typing_extensions import Literal

CalibrationMethod = Literal["isotonic", "sigmoid"]
MetricComponents = List[str]
Details = Dict[str, float]