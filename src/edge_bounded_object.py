"""Edge detection class for image processing."""

from typing import Union, List
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import numpy as np
import cv2

from .config import Config


class EdgeDetectionMethod(Enum):
    """Enum for edge detection methods."""
    CANNY = "canny"
    SOBEL = "sobel"
    DEXI = "dexi"  # future — placeholder stub

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

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured edge detection algorithm to produce an edge map.

        Args:
            image: 2D greyscale numpy array (output of blur stage).

        Returns:
            np.ndarray: Binary edge map of the same spatial size.

        Raises:
            NotImplementedError: If edge_detection_method is 'dexi' (not yet implemented).
            ValueError: If edge_detection_method is not a recognised value.
        """
        method = self.config.edge_detection_method

        if method == EdgeDetectionMethod.CANNY.value:
            low, high = self.config.canny_threshold
            return cv2.Canny(image, low, high)

        elif method == EdgeDetectionMethod.SOBEL.value:
            ksize = self.config.sobel_kernel_size
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            return np.clip(magnitude, 0, 255).astype(np.uint8)

        elif method == EdgeDetectionMethod.DEXI.value:
            raise NotImplementedError("DEXI edge detection is not yet implemented")

        else:
            raise ValueError(
                f"Unsupported edge detection method: '{method}'. "
                f"Expected one of: {[m.value for m in EdgeDetectionMethod]}"
            )
