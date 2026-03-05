"""Configuration classes for image processing operations."""

import json
from pathlib import Path
from typing import List
import urllib.request
import urllib.error


class Config:
    def __init__(
        self,
        hardware_accelerated: bool = False,
        fattened_size_offset: int = 0,
        white_threshold: int = 200,
        border_blur_size: int = 20,
        edge_detection_method: str = "canny",
        canny_threshold: List[int] = None,
        sobel_kernel_size: int = 3,
        dilation_size: int = 3,
        grouping_proximity: int = 10,
    ) -> None:
        self.hardware_accelerated = hardware_accelerated
        self.fattened_size_offset = fattened_size_offset
        self.white_threshold = white_threshold
        self.border_blur_size = border_blur_size
        self.edge_detection_method = edge_detection_method
        self.canny_threshold = canny_threshold if canny_threshold is not None else [50, 150]
        self.sobel_kernel_size = sobel_kernel_size
        self.dilation_size = dilation_size
        self.grouping_proximity = grouping_proximity

    @classmethod
    def from_json(cls, json_source):
        """Deserialize a Config object from a JSON file or URL.

        Args:
            json_source: Path to JSON file (str or Path) or URL (str)

        Returns:
            Config: A new Config instance with properties loaded from JSON

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the JSON is invalid or not a dict
            urllib.error.URLError: If the URL cannot be accessed
            json.JSONDecodeError: If the JSON content is malformed
        """
        # Try to determine if it's a URL or file path
        if isinstance(json_source, str) and (json_source.startswith('http://') or json_source.startswith('https://')):
            return cls._from_json_url(json_source)
        else:
            return cls._from_json_file(json_source)

    @classmethod
    def _from_json_file(cls, json_file_path):
        """Load config from a local JSON file."""
        file_path = Path(json_file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"JSON config file not found: {file_path}")

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in config file: {e}")

        return cls._create_from_dict(data)

    @classmethod
    def _from_json_url(cls, json_url):
        """Load config from a JSON URL."""
        try:
            with urllib.request.urlopen(json_url) as response:
                data = json.loads(response.read().decode('utf-8'))
        except urllib.error.URLError as e:
            raise urllib.error.URLError(f"Failed to fetch config from URL {json_url}: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from URL {json_url}: {e}")

        return cls._create_from_dict(data)

    @classmethod
    def _create_from_dict(cls, data):
        """Create Config instance from dictionary."""
        if not isinstance(data, dict):
            raise ValueError("JSON config must be a dictionary/object")

        # Extract parameters with defaults
        hardware_accelerated = data.get('hardware_accelerated', False)
        fattened_size_offset = data.get('fattened_size_offset', 0)
        white_threshold = data.get('white_threshold', 200)
        border_blur_size = data.get('border_blur_size', 20)
        edge_detection_method = data.get('edge_detection_method', 'canny')
        canny_threshold = data.get('canny_threshold', [50, 150])
        sobel_kernel_size = data.get('sobel_kernel_size', 3)
        dilation_size = data.get('dilation_size', 3)
        grouping_proximity = data.get('grouping_proximity', 10)

        return cls(
            hardware_accelerated=hardware_accelerated,
            fattened_size_offset=fattened_size_offset,
            white_threshold=white_threshold,
            border_blur_size=border_blur_size,
            edge_detection_method=edge_detection_method,
            canny_threshold=canny_threshold,
            sobel_kernel_size=sobel_kernel_size,
            dilation_size=dilation_size,
            grouping_proximity=grouping_proximity,
        )
