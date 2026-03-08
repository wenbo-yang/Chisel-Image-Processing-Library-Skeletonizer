"""Morphological dilation helper for edge gap closure."""

import numpy as np
import cv2

from ...config import Config
from ..processor import Processor


class EdgeCloser(Processor):
    """Close small gaps in edge fragments using dilation."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config

    def apply(self, edge_region: np.ndarray) -> np.ndarray:
        """Dilate `edge_region` and return the bridged edge map."""
        dilation_size = self.config.dilation_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )
        return cv2.dilate(edge_region, kernel, iterations=1)

    # Backwards-compatible alias
    def close(self, edge_region: np.ndarray) -> np.ndarray:
        return self.apply(edge_region)
