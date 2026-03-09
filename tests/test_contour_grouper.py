"""Tests for ContourGrouper processor."""

import pytest
from src.config import Config
from src.processor.contour_grouper import ContourGrouper


class TestContourGrouper:
    """Test suite for ContourGrouper processor."""

    @pytest.fixture
    def config(self):
        """Get a default Config instance."""
        return Config()

    def test_contour_grouper_init(self, config):
        """Test ContourGrouper initialization with valid config."""
        grouper = ContourGrouper(config)
        assert isinstance(grouper, ContourGrouper)
        assert grouper.config == config
