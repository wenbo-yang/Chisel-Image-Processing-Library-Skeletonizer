from typing import Union, List
from pathlib import Path
import numpy as np

from .config import Config

class Extractor:
    """Extractor class for extracting object from images or videos."""

    def __init__(self, config: Config) -> None:
        if not isinstance(config, Config):
            raise TypeError(f"config must be a Config instance, got {type(config)}")

        self.config = config

    def extract_from_image(self, image: Union[np.ndarray, List[bytes], str, Path]) -> List[np.ndarray]:
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

    def _extract_bitmap(self, image: np.ndarray) -> List[np.ndarray]:
        pass

    def _extract_byte_array_2d(self, image: List[bytes]) -> List[np.ndarray]:
        pass

    def _extract_file(self, image: Union[str, Path]) -> List[np.ndarray]:
        pass

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

