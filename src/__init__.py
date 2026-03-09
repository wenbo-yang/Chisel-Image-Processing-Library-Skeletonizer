"""Chisel Image Processing Library"""

from .config import Config
from .processor.skeletonizer import Thinning
from .extractor import Extractor
from .models.edge_bounded_object import EdgeBoundedObject, EdgeDetectionMethod, CannyData, SobelData
from .processor.edge_detector import EdgeDetector
from .processor.blurer import ImageBlur
from .processor.contour_grouper import ContourGrouper
from .processor.edge_connector import EdgeConnector

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
