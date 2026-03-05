"""Size-adaptive Gaussian blur for image preprocessing."""

import numpy as np
import cv2

from .config import Config


class ImageBlur:
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

        Kernel size formula (when config.border_blur_size == 0):
            kernel_size = max(3, min(height, width) // 20) | 1  # forced odd

        If config.border_blur_size > 0, that value is used directly as the
        kernel size (forced odd via bitwise OR with 1).

        Args:
            image: 2D greyscale numpy array.

        Returns:
            np.ndarray: Blurred image of the same shape and dtype.
        """
        if self.config.border_blur_size > 0:
            kernel_size = self.config.border_blur_size | 1
        else:
            kernel_size = max(3, min(image.shape[0], image.shape[1]) // 20) | 1

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
