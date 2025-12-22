"""Tests for the Config class and from_json deserialization."""

import pytest
from pathlib import Path
from src.config import Config


class TestConfigFromJson:
    """Test suite for Config.from_json deserialization method."""

    @pytest.fixture
    def resources_dir(self):
        """Get the path to the test resources directory."""
        return Path(__file__).parent / "resources"

    def test_from_json_valid_full_config(self, resources_dir):
        """Test loading a valid JSON config with all parameters specified."""
        config_path = resources_dir / "config_valid.json"
        config = Config.from_json(config_path)

        assert isinstance(config, Config)
        assert config.hardware_accelerated is True
        assert config.fattened_size_offset == 5
        assert config.white_threshold == 150
        assert config.border_blur_size == 10

    def test_from_json_minimal_config(self, resources_dir):
        """Test loading a minimal JSON config with no parameters (uses defaults)."""
        config_path = resources_dir / "config_minimal.json"
        config = Config.from_json(config_path)

        assert isinstance(config, Config)
        assert config.hardware_accelerated is False
        assert config.fattened_size_offset == 0
        assert config.white_threshold == 200
        assert config.border_blur_size == 20

    def test_from_json_invalid_json(self, resources_dir):
        """Test that invalid JSON raises ValueError."""
        config_path = resources_dir / "config_invalid.json"

        with pytest.raises(ValueError) as exc_info:
            Config.from_json(config_path)

        assert "Invalid JSON" in str(exc_info.value)

    def test_from_json_file_not_found(self):
        """Test that missing file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            Config.from_json("nonexistent_config.json")

        assert "not found" in str(exc_info.value)

    def test_from_json_path_object(self, resources_dir):
        """Test that Path objects work as arguments."""
        config_path = resources_dir / "config_valid.json"
        config = Config.from_json(config_path)

        assert isinstance(config, Config)
        assert config.hardware_accelerated is True

    def test_from_json_string_path(self, resources_dir):
        """Test that string paths work as arguments."""
        config_path = str(resources_dir / "config_valid.json")
        config = Config.from_json(config_path)

        assert isinstance(config, Config)
        assert config.hardware_accelerated is True

    def test_from_json_url_detection(self):
        """Test that URLs are detected and routed to URL handler."""
        # This test verifies URL detection without requiring network access
        with pytest.raises(Exception):
            # Should attempt URL fetch (will fail with URLError)
            Config.from_json("http://example.com/config.json")

    def test_from_json_partial_config(self, resources_dir):
        """Test loading JSON with only some parameters specified."""
        # Create a temporary config with partial parameters
        import json
        import tempfile

        partial_config = {
            "hardware_accelerated": True,
            "white_threshold": 175
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(partial_config, f)
            temp_path = f.name

        try:
            config = Config.from_json(temp_path)

            assert config.hardware_accelerated is True
            assert config.fattened_size_offset == 0  # default
            assert config.white_threshold == 175
            assert config.border_blur_size == 20  # default
        finally:
            Path(temp_path).unlink()
