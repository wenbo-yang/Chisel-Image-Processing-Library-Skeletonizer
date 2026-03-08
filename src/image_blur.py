"""Gaussian blur for image preprocessing.

Current behaviour: `ImageBlur.apply()` uses `Config.border_blur_size` directly as the
Gaussian kernel size (forced odd via bitwise OR). `Config` also exposes
`blur_kernel_min` and `blur_kernel_max` but they are not used by the implementation.

This module provides a simple in-memory blur step before edge detection.
"""

import numpy as np
import cv2

from .config import Config
from .processor.process import Processor


class ImageBlur(Processor):
    """Applies Gaussian blur to an image before edge detection.

    The kernel size scales adaptively with the image dimensions so that
    blur strength is proportional regardless of resolution. The computed
    kernel can be overridden by setting config.border_blur_size > 0.
    """

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")
        self.config = config

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Gaussian blur to a 2D greyscale image.

        If `self.config.border_blur_size > 0`, that odd integer is used as the
        kernel size. Otherwise the kernel is interpolated between
        `self.config.blur_kernel_min` and `self.config.blur_kernel_max` based on
        image size relative to `self.config.blur_dim_min` /
        `self.config.blur_dim_max`. The kernel is always forced odd.

        Args:
            image: 2D greyscale numpy array.

        Returns:
            np.ndarray: Blurred image of the same shape and dtype.
        """
        # Explicit override if set in config
        if self.config.border_blur_size > 0:
            kernel_size = int(self.config.border_blur_size) | 1
        else:
            # Use linear scaling based on the larger image dimension
            max_dim = max(image.shape[0], image.shape[1])

            # Fixed thresholds (pixels) — now configurable via Config
            low_dim = self.config.blur_dim_min
            high_dim = self.config.blur_dim_max

            # Read configured minimum/maximum kernel sizes
            k_min = self.config.blur_kernel_min
            k_max = self.config.blur_kernel_max

            # No validation here; values are assumed valid integers from Config

            if max_dim <= low_dim:
                kernel_size = k_min
            elif max_dim >= high_dim:
                kernel_size = k_max
            else:
                t = (max_dim - low_dim) / float(high_dim - low_dim)
                kernel_f = k_min + t * (k_max - k_min)
                kernel_size = int(round(kernel_f))

            # force odd kernel size
            kernel_size = kernel_size | 1

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
