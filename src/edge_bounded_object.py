"""Edge detection class for image processing."""

from typing import Union, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .config import Config


class EdgeDetectionMethod(Enum):
    """Enum for edge detection methods."""
    CANNY = "canny"
    SOBEL = "sobel"


@dataclass
class CannyData:
    """Data class for Canny edge detection parameters."""
    line_threshold: List[int]  # Array of 2 numbers [min, max]

@dataclass
class SobelData:
    """Data class for Sobel edge detection parameters."""
    pass


@dataclass
class EdgeBoundedObject:
    """Data class for edge bounded object parameters."""
    edge_detection_method: EdgeDetectionMethod
    blur_strength: List[int]  # Array of 2 numbers [min, max], range 3-20
    edge_data_description: Union[CannyData, SobelData]
    coordinates: List[tuple]  # List of (x, y) coordinates in the original image
    bounded_image: np.ndarray  # Bounded image extracted from coordinates

    def __post_init__(self):
        """Validate blur_strength is within acceptable range."""
        if len(self.blur_strength) != 2:
            raise ValueError("blur_strength must be an array of 2 numbers")
        if self.blur_strength[0] < 3 or self.blur_strength[1] > 20:
            raise ValueError("blur_strength values must be between 3 and 20")
        if self.blur_strength[0] > self.blur_strength[1]:
            raise ValueError("blur_strength min must be <= max")


class EdgeDetector:
    """EdgeDetector class for detecting edges in images."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config
