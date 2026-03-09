"""Edge detector processor (Canny / Sobel)."""
from __future__ import annotations

from typing import Union, List
import numpy as np
import cv2

from ..config import Config
from .processor import Processor
# Avoid importing model datatypes here to prevent circular imports; use
# simple string values from config for method selection.


class EdgeDetector(Processor):
    """Detect edges using the configured method (Canny or Sobel)."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Return an edge map for `image` based on `config` settings."""

        method = self.config.edge_detection_method

        if method == "canny":
            low, high = self.config.canny_threshold
            return cv2.Canny(image, low, high)

        if method == "sobel":
            ksize = self.config.sobel_kernel_size
            sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
            sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
            magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
            return np.clip(magnitude, 0, 255).astype(np.uint8)

        if method == "dexi":
            raise NotImplementedError("DEXI edge detection is not yet implemented")

        raise ValueError(f"Unsupported edge detection method: '{method}'")

    def detect(self, image: np.ndarray) -> np.ndarray:
        """Backward-compatible alias for `apply`."""
        return self.apply(image)
