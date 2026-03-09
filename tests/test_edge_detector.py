"""Tests for EdgeDetector processor."""

import pytest
from src.config import Config
from src.processor.edge_detector import EdgeDetector


class TestEdgeDetector:
    """Test suite for EdgeDetector processor."""

    @pytest.fixture
    def config(self):
        """Get a default Config instance."""
        return Config()

    def test_edge_detector_init(self, config):
        """Test EdgeDetector initialization with valid config."""
        detector = EdgeDetector(config)
        assert isinstance(detector, EdgeDetector)
        assert detector.config == config
