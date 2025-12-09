"""Zhang-Suen thinning algorithm wrapper for image skeletonization."""

from typing import Union, List
from pathlib import Path
import numpy as np
import cv2
import cv2.ximgproc

from .config import Config

class Thinning:    
    def __init__(self, config: Config) -> None:
        self.config = config
    
    def skeletonize(self, image_input: Union[np.ndarray, List[bytes], str, Path]) -> np.ndarray:
        # Dispatch to appropriate method based on input type
        if isinstance(image_input, np.ndarray):
            return self._skeletonize_bitmap(image_input)
        elif isinstance(image_input, list):
            return self._skeletonize_byte_array_2d(image_input)
        elif isinstance(image_input, (str, Path)):
            return self._skeletonize_file(image_input)
        else:
            raise TypeError(
                f"Unsupported image_input type: {type(image_input)}. "
                "Expected np.ndarray, List[bytes], str, or Path."
            )
    
    def _skeletonize_bitmap(self, bitmap: np.ndarray) -> np.ndarray:
        if not isinstance(bitmap, np.ndarray):
            raise ValueError("Bitmap must be a numpy ndarray")
        
        if bitmap.ndim != 2:
            raise ValueError(
                f"Bitmap must be 2D, got {bitmap.ndim}D array"
            )
        
        if bitmap.size == 0:
            raise ValueError("Bitmap cannot be empty")
        
        # Ensure the image is binary (0 or 255)
        # Pixels below white threshold become 0 (black), above become 255 (white)
        binary_image = np.where(bitmap < self.config.white_threshold, 255, 0).astype(np.uint8)
        
        # Apply Zhang-Suen thinning
        skeletonized = cv2.ximgproc.thinning(
            binary_image,
            thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
        )
        
        return self._fatten_skeleton(skeletonized)
    
    def _fatten_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Fatten the skeletonized image based on config.fattened_size_offset.
        
        Args:
            skeleton: The skeletonized binary image
        
        Returns:
            np.ndarray: The fattened skeleton image, or original if fattened_size_offset is 0
        """
        if self.config.fattened_size_offset == 0:
            return skeleton
        
        # Create a kernel for dilation with size 1 + fattened_size_offset
        kernel_size = 1 + self.config.fattened_size_offset
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # Dilate the skeleton to fatten it
        fattened = cv2.dilate(skeleton, kernel, iterations=1)
        
        return fattened
    
    def _skeletonize_png_bytes(self, png_bytes: bytes) -> np.ndarray:

        # Decode the PNG byte array to an image
        nparr = np.frombuffer(png_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(
                "Failed to decode byte array as PNG image. "
                "Ensure it's a valid encoded PNG format."
            )
        
        # Use bitmap skeletonization on the decoded image
        return self._skeletonize_bitmap(image)
    
    def _skeletonize_byte_array_2d(self, byte_array_2d: List[bytes]) -> np.ndarray:
        if not isinstance(byte_array_2d, list):
            raise ValueError("Input must be a list of bytes objects")
        
        if len(byte_array_2d) == 0:
            raise ValueError("Byte array list cannot be empty")
        
        # Convert list of byte arrays to numpy array
        try:
            image_array = np.array(
                [np.frombuffer(row, dtype=np.uint8) for row in byte_array_2d],
                dtype=np.uint8
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to convert 2D byte array to image. "
                f"Ensure each element is a bytes object: {e}"
            )
        
        if image_array.size == 0:
            raise ValueError("Converted image array is empty")
        
        # Use bitmap skeletonization on the converted image
        return self._skeletonize_bitmap(image_array)
    
    def _skeletonize_file(self, file_path: Union[str, Path]) -> np.ndarray:
        """Skeletonize an image from a file.
        
        Automatically detects the file type based on extension and reads
        the file accordingly. Supports common image formats (PNG, JPG, BMP, etc.).
        
        Args:
            file_path: Path to the image file (str or Path object)
        
        Returns:
            np.ndarray: Skeletonized binary image
        
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file cannot be read as a valid image
            ValueError: If the file type is not supported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Get file extension
        file_ext = file_path.suffix.lower()
        
        # Supported image formats
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}
        
        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
        
        # Read the image in grayscale
        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            raise ValueError(
                f"Failed to read image file: {file_path}. "
                f"Ensure it's a valid {file_ext} image format."
            )
        
        # Use bitmap skeletonization on the loaded image
        return self._skeletonize_bitmap(image)
