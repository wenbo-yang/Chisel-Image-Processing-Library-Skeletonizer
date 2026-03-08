from typing import Union, List
from pathlib import Path
import numpy as np
import cv2

from .config import Config
from .blurer import ImageBlur
from .edge_bounded_object import EdgeDetector, EdgeBoundedObject, EdgeDetectionMethod, CannyData, SobelData
from .contour_grouper import ContourGrouper
from .edge_connector import EdgeConnector

class Extractor:
    """Extractor class for extracting object from images or videos."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")

        self.config = config

    def extract_from_image(self, image: Union[np.ndarray, List[bytes], str, Path]) -> List[EdgeBoundedObject]:
        if isinstance(image, np.ndarray):
            return self._extract_bitmap(image)
        elif isinstance(image, list):
            return self._extract_byte_array_2d(image)
        elif isinstance(image, (str, Path)):
            return self._extract_file(image)
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected np.ndarray, List[bytes], str, or Path."
            )

    def _extract_bitmap(self, image: np.ndarray) -> List[EdgeBoundedObject]:
        """Core extraction pipeline. Validates the array, then runs all stages in order."""
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be a numpy ndarray")
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got {image.ndim}D array")
        if image.size == 0:
            raise ValueError("Image cannot be empty")

        # Stage 2: size-adaptive Gaussian blur (result cached for inspection)
        blurred = ImageBlur(self.config).apply(image)
        self._last_blurred = blurred

        # Stage 3: edge detection
        edge_map = EdgeDetector(self.config).apply(blurred)

        # Stage 4: group contours by proximity
        groups = ContourGrouper(self.config).apply(edge_map)

        # Compute blur_strength metadata (clamped to EdgeBoundedObject valid range [3, 19])
        if self.config.border_blur_size > 0:
            k = self.config.border_blur_size | 1
        else:
            k = max(3, min(image.shape[0], image.shape[1]) // 20) | 1
        k = max(3, min(k, 19))
        blur_strength = [k, k]

        method = EdgeDetectionMethod(self.config.edge_detection_method)
        if self.config.edge_detection_method == EdgeDetectionMethod.CANNY.value:
            low, high = self.config.canny_threshold
            edge_data = CannyData(line_threshold=[low, high])
        else:
            edge_data = SobelData()

        # Stage 5 + 6: close edges and build output objects
        edge_connector = EdgeConnector(self.config)
        results = []

        for group in groups:
            all_points = np.vstack(group)
            x, y, w, h = cv2.boundingRect(all_points)

            # Close edge gaps within the bounding region
            edge_region = edge_map[y:y + h, x:x + w]
            edge_connector.apply(edge_region)

            # Crop from the original (pre-blur) image to preserve full detail
            bounded_image = image[y:y + h, x:x + w].copy()

            # Translate contour coordinates to original image space
            coordinates = [
                (int(pt[0][0]), int(pt[0][1]))
                for c in group
                for pt in c
            ]

            results.append(EdgeBoundedObject(
                edge_detection_method=method,
                blur_strength=blur_strength,
                edge_data_description=edge_data,
                coordinates=coordinates,
                bounded_image=bounded_image,
            ))

        return results

    def _extract_byte_array_2d(self, image: List[bytes]) -> List[EdgeBoundedObject]:
        """Convert a list of byte rows into a 2D array and delegate to _extract_bitmap."""
        if not isinstance(image, list):
            raise ValueError("Input must be a list of bytes objects")
        if len(image) == 0:
            raise ValueError("Byte array list cannot be empty")

        try:
            image_array = np.array(
                [np.frombuffer(row, dtype=np.uint8) for row in image],
                dtype=np.uint8,
            )
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Failed to convert 2D byte array to image. "
                f"Ensure each element is a bytes object: {e}"
            )

        if image_array.size == 0:
            raise ValueError("Converted image array is empty")

        return self._extract_bitmap(image_array)

    def _extract_file(self, image: Union[str, Path]) -> List[EdgeBoundedObject]:
        """Read a greyscale image from a file path and delegate to _extract_bitmap."""
        file_path = Path(image)

        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")

        file_ext = file_path.suffix.lower()
        supported_formats = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif'}

        if file_ext not in supported_formats:
            raise ValueError(
                f"Unsupported file type: {file_ext}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )

        img = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(
                f"Failed to read image file: {file_path}. "
                f"Ensure it's a valid {file_ext} image format."
            )

        return self._extract_bitmap(img)

    def extract_from_video(self, video_input: Union[str, None]) -> List[np.ndarray]:
        if video_input is None:
            return self._extract_camera()
        elif video_input.startswith("http://") or video_input.startswith("https://"):
            # Check if it's an image sequence URL or video URL
            if any(ext in video_input.lower() for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif']):
                return self._extract_image_sequence_url(video_input)
            else:
                return self._extract_video_url(video_input)
        else:
            # Assume it's a base64 encoded string
            return self._extract_base64_video(video_input)

    def _extract_camera(self) -> List[np.ndarray]:
        pass

    def _extract_video_url(self, video_url: str) -> List[np.ndarray]:
        pass

    def _extract_image_sequence_url(self, image_sequence_url: str) -> List[np.ndarray]:
        pass

    def _extract_base64_video(self, base64_string: str) -> List[np.ndarray]:
        pass

