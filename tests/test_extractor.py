import pytest
from src.config import Config
from src.extractor import Extractor


class TestExtractor:
    """Test suite for the Extractor class."""

    def test_extractor_init_valid_config(self):
        """Test Extractor initialization with a valid Config object."""
        config = Config()
        extractor = Extractor(config)

        assert isinstance(extractor, Extractor)
        assert extractor.config is config

