"""Morphological edge closure to bridge disconnected edge fragments."""

import numpy as np
import cv2

from .config import Config


class EdgeCloser:
    """Closes gaps in edge fragments using morphological dilation.

    Applies an elliptical dilation kernel of size config.dilation_size
    to bridge small gaps between disconnected edges, producing a more
    complete object boundary before the final crop is extracted.
    """

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config

    def close(self, edge_region: np.ndarray) -> np.ndarray:
        """Apply dilation to close gaps in an edge region.

        Args:
            edge_region: 2D binary numpy array of edge fragments.

        Returns:
            np.ndarray: Edge region with gaps bridged by dilation.
        """
        dilation_size = self.config.dilation_size
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilation_size, dilation_size)
        )
        return cv2.dilate(edge_region, kernel, iterations=1)
