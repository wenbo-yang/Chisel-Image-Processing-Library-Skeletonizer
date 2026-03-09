"""Edge detection helpers and data classes."""

from typing import Union, List
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..config import Config
from ..processor.edge_detector import EdgeDetector


class EdgeDetectionMethod(Enum):
    """Enum for edge detection methods."""
    CANNY = "canny"
    SOBEL = "sobel"
    DEXI = "dexi"  # future — placeholder stub


@dataclass
class CannyData:
    """Canny edge detection parameters."""
    line_threshold: List[int]


@dataclass
class SobelData:
    """Sobel edge detection parameters (placeholder)."""
    pass

@dataclass
class EdgeBoundedObject:
    """Container for an extracted object bounded by edges."""
    edge_detection_method: EdgeDetectionMethod
    blur_strength: List[int]
    edge_data_description: Union[CannyData, SobelData]
    coordinates: List[tuple]
    bounded_image: np.ndarray

    def __post_init__(self):
        if len(self.blur_strength) != 2:
            raise ValueError("blur_strength must be an array of 2 numbers")
        if self.blur_strength[0] < 3 or self.blur_strength[1] > 20:
            raise ValueError("blur_strength values must be between 3 and 20")
        if self.blur_strength[0] > self.blur_strength[1]:
            raise ValueError("blur_strength min must be <= max")

