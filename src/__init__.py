"""Chisel Image Processing Library"""

from .config import Config
from .thinning import Thinning
from .extractor import Extractor
from .edge_bounded_object import EdgeBoundedObject, EdgeDetector, EdgeDetectionMethod, CannyData, SobelData
from .image_blur import ImageBlur
from .contour_grouper import ContourGrouper
from .edge_closer import EdgeCloser

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
    "EdgeCloser",
]
