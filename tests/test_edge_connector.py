"""Tests for EdgeConnector processor."""

import pytest
from src.config import Config
from src.processor.edge_connector import EdgeConnector


class TestEdgeConnector:
    """Test suite for EdgeConnector processor."""

    @pytest.fixture
    def config(self):
        """Get a default Config instance."""
        return Config()

    def test_edge_connector_init(self, config):
        """Test EdgeConnector initialization with valid config."""
        connector = EdgeConnector(config)
        assert isinstance(connector, EdgeConnector)
        assert connector.config == config
