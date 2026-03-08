"""Chisel Image Processing Library"""

from .config import Config
from .skeletonizer import Thinning
from .extractor import Extractor
from .edge_bounded_object import EdgeBoundedObject, EdgeDetector, EdgeDetectionMethod, CannyData, SobelData
from .blurer import ImageBlur
from .contour_grouper import ContourGrouper
from .edge_connector import EdgeConnector

__version__ = "0.1.0"
__author__ = "Your Name"
__description__ = "A Python library for image processing with skeletonization capabilities"

__all__ = [
    "Config",
    "Thinning",
    "Extractor",
    "EdgeBoundedObject",
    "EdgeDetector",
    "EdgeDetectionMethod",
    "CannyData",
    "SobelData",
    "ImageBlur",
    "ContourGrouper",
    "EdgeConnector",
]
