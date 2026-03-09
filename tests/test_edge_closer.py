"""Tests for EdgeCloser processor."""

import pytest
from src.config import Config
from src.processor.edge_closer import EdgeCloser


class TestEdgeCloser:
    """Test suite for EdgeCloser processor."""

    @pytest.fixture
    def config(self):
        """Get a default Config instance."""
        return Config()

    def test_edge_closer_init(self, config):
        """Test EdgeCloser initialization with valid config."""
        closer = EdgeCloser(config)
        assert isinstance(closer, EdgeCloser)
        assert closer.config == config
