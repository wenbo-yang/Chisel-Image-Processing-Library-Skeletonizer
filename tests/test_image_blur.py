"""Tests for ImageBlur processor."""

import pytest
from src.config import Config
from src.processor.blurer import ImageBlur


class TestImageBlur:
    """Test suite for ImageBlur processor."""

    @pytest.fixture
    def config(self):
        """Get a default Config instance."""
        return Config()

    def test_image_blur_init(self, config):
        """Test ImageBlur initialization with valid config."""
        blurrer = ImageBlur(config)
        assert isinstance(blurrer, ImageBlur)
        assert blurrer.config == config
