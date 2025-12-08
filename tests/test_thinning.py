"""Tests for the Thinning class and skeletonize function."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.thinning import Thinning
from src.config import Config


class TestThinningSkeletonize:
    """Test suite for the Thinning.skeletonize method."""
    
    @pytest.fixture
    def thinning_instance(self):
        """Create a Thinning instance with default configuration."""
        config = Config(hardware_accelerated=False)
        return Thinning(config)
    
    @pytest.fixture
    def resources_dir(self):
        """Get the path to the test resources directory."""
        return Path(__file__).parent / "resources"
    
    @pytest.fixture
    def temp_dir(self):
        """Create and get the path to the temp directory."""
        temp_path = Path(__file__).parent / "temp"
        temp_path.mkdir(exist_ok=True)
        return temp_path
    
    def test_skeletonize_silhouette_fat_man_running(self, thinning_instance, resources_dir, temp_dir):
        """Test skeletonization of fat man running silhouette image.
        
        Loads a PNG file and skeletonizes it. Asserts that the output
        is not completely white (i.e., contains some black pixels).
        Saves the output bitmap to the temp folder.
        """
        image_path = resources_dir / "silhouette_fat_man_running.png"
        
        # Skeletonize the image
        result = thinning_instance.skeletonize(str(image_path))
        
        # Assert output is a numpy array
        assert isinstance(result, np.ndarray)
        
        # Assert output is 2D
        assert result.ndim == 2
        
        # Assert output is not all white (contains some non-255 values)
        assert not np.all(result == 255), "Output should not be all white"
        
        # Assert output contains some black pixels (0 values)
        assert np.any(result == 0), "Output should contain black pixels"
        
        # Save output bitmap to temp folder
        output_path = temp_dir / "silhouette_fat_man_running_output.png"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), "Output bitmap was not saved"
    
    def test_skeletonize_silhouette_man_running(self, thinning_instance, resources_dir, temp_dir):
        """Test skeletonization of man running silhouette image.
        
        Loads a PNG file and skeletonizes it. Asserts that the output
        is not completely white (i.e., contains some black pixels).
        Saves the output bitmap to the temp folder.
        """
        image_path = resources_dir / "silhouette_man_running.png"
        
        # Skeletonize the image
        result = thinning_instance.skeletonize(str(image_path))
        
        # Assert output is a numpy array
        assert isinstance(result, np.ndarray)
        
        # Assert output is 2D
        assert result.ndim == 2
        
        # Assert output is not all white (contains some non-255 values)
        assert not np.all(result == 255), "Output should not be all white"
        
        # Assert output contains some black pixels (0 values)
        assert np.any(result == 0), "Output should contain black pixels"
        
        # Save output bitmap to temp folder
        output_path = temp_dir / "silhouette_man_running_output.png"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), "Output bitmap was not saved"
    
    def test_skeletonize_2d_byte_array_colors(self, thinning_instance, resources_dir, temp_dir):
        """Test skeletonization using 2D byte array with inverted colors.
        
        Reads silhouette_man_running.png, converts it to a 2D byte array with
        0 representing white and 255 representing black, then feeds it into
        the skeletonize function. Asserts that the output is not all white.
        Saves the output bitmap to the temp folder.
        """
        image_path = resources_dir / "silhouette_man_running.png"
        
        # Read the image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to read image file"
        
        # Convert 2D array to 2D byte array (list of byte arrays)
        byte_array_2d = [bytes(row) for row in image]
        
        # Skeletonize the 2D byte array
        result = thinning_instance.skeletonize(byte_array_2d)
        
        # Assert output is a numpy array
        assert isinstance(result, np.ndarray)
        
        # Assert output is 2D
        assert result.ndim == 2
        
        # Assert output is not all white (contains some non-255 values)
        assert not np.all(result == 255), "Output should not be all white"
        
        # Assert output contains some black pixels (0 values)
        assert np.any(result == 0), "Output should contain black pixels"
        
        # Save output bitmap to temp folder
        output_path = temp_dir / "silhouette_man_running_converted_output.png"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), "Output bitmap was not saved"
