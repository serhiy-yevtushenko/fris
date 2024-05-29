"""Common types definitions."""

from typing import Callable
from typing import Sequence
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

DataPoint: TypeAlias = NDArray[np.float64]
CachedDistanceMatrix: TypeAlias = NDArray[np.float64]
DistanceFunctionType = Callable[[DataPoint, DataPoint], float]
ClassType = int | str
DataPointArray: TypeAlias = NDArray[np.float64]
DataPointList = list[DataPoint]
LabelsArray = Sequence[ClassType]
