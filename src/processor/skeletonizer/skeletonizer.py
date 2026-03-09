"""Zhang-Suen thinning (skeletonization) helper."""

import numpy as np
import cv2
import cv2.ximgproc

from ...config import Config
from ..processor import Processor

class Thinning(Processor):
    def __init__(self, config: Config) -> None:
        self.config = config

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Apply Zhang-Suen thinning to a 2D grayscale image array.
        
        Background color is automatically determined from pixel at (0,0).
        If pixel value > 127, background is white; otherwise, background is black.
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"image must be np.ndarray, got {type(image)}")
        if image.ndim != 2:
            raise ValueError(f"image must be 2D, got {image.ndim}D array")
        if image.size == 0:
            raise ValueError("image cannot be empty")
        
        # Determine background color from corner pixel (0,0)
        is_background_white = image[0, 0] > 127
        
        return self._thin_bitmap(image, is_background_white)

    def thin(self, image: np.ndarray) -> np.ndarray:
        """Alias for apply()."""
        return self.apply(image)

    def _thin_bitmap(self, bitmap: np.ndarray, is_background_white: bool) -> np.ndarray:

        # If background is white, invert the image to black background
        image_to_process = bitmap
        if not is_background_white:
            image_to_process = 255 - bitmap

        # Apply border blur if configured
        if self.config.border_blur_size > 0:
            image_to_process = self._blur_border(image_to_process)

        # Ensure the image is binary (0 or 255)
        # Pixels below white threshold become 0 (black), above become 255 (white)
        binary_image = np.where(image_to_process < self.config.white_threshold, 255, 0).astype(np.uint8)

        # Apply Zhang-Suen thinning
        skeletonized = cv2.ximgproc.thinning(
            binary_image,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )

        # Invert output if background is white (i.e., if is_background_white is True)
        output = skeletonized
        if is_background_white:
            output = 255 - skeletonized

        return self._fatten_skeleton(output)

    def _fatten_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        if self.config.fattened_size_offset == 0:
            return skeleton

        # Create a kernel for dilation with size 1 + fattened_size_offset
        kernel_size = 1 + self.config.fattened_size_offset
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

        # To fatten a black skeleton on white background, we need to invert, dilate, then invert back
        # This grows the black pixels (skeleton) rather than the white background
        inverted_skeleton = 255 - skeleton
        dilated_inverted = cv2.dilate(inverted_skeleton, kernel, iterations=1)
        fattened = 255 - dilated_inverted

        return fattened

    def _blur_border(self, image: np.ndarray) -> np.ndarray:
        if self.config.border_blur_size <= 0:
            return image

        # Define the border region
        border_size = self.config.border_blur_size

        # Apply Gaussian blur to the entire image
        blurred = cv2.GaussianBlur(image, (border_size | 1, border_size | 1), 0)

        return blurred
