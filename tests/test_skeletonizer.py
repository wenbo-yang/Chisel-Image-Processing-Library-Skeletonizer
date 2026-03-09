"""Tests for the Thinning class and thin function."""

import pytest
import numpy as np
import cv2
from pathlib import Path
from src.processor.skeletonizer import Thinning
from src.config import Config


class TestThinningSkeletonize:
    # Test suite for the Thinning.thin method.

    @pytest.fixture
    def thinning_instance(self):
        # Create a Thinning instance with default configuration.
        config = Config(hardware_accelerated=False)
        return Thinning(config)

    @pytest.fixture
    def resources_dir(self):
        # Get the path to the test resources directory.
        return Path(__file__).parent / "resources"

    @pytest.fixture
    def temp_dir(self):
        # Create and get the path to the temp directory.
        temp_path = Path(__file__).parent / "temp"
        temp_path.mkdir(exist_ok=True)
        return temp_path

    def test_skeletonize_silhouette_fat_man_running(self, thinning_instance, resources_dir, temp_dir):
        # Test skeletonization of fat man running silhouette image.
        # Loads a PNG file and skeletonizes it. Asserts that the output
        # is not completely white (i.e., contains some black pixels).
        # Saves the output bitmap to the temp folder.
        image_path = resources_dir / "silhouette_fat_man_running.png"

        # Load image as numpy array
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to read image file"

        # Skeletonize the image
        result = thinning_instance.thin(image)

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
        # Test skeletonization of man running silhouette image.
        # Loads a PNG file and skeletonizes it. Asserts that the output
        # is not completely white (i.e., contains some black pixels).
        # Saves the output bitmap to the temp folder.
        image_path = resources_dir / "silhouette_man_running.png"

        # Load image as numpy array
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to read image file"

        # Skeletonize the image
        result = thinning_instance.thin(image)

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
        # Test skeletonization using 2D byte array with inverted colors.
        # Reads silhouette_man_running.png, converts it to a numpy array,
        # then feeds it into the skeletonize function. Asserts that the output is not all white.
        # Saves the output bitmap to the temp folder.
        image_path = resources_dir / "silhouette_man_running.png"

        # Read the image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to read image file"

        # Skeletonize the numpy array
        result = thinning_instance.thin(image)

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

    def test_skeletonize_inverted_colors(self, thinning_instance, resources_dir, temp_dir):
        # Test skeletonization with inverted image colors.
        # Reads silhouette_man_running.png, inverts the colors (white background becomes black),
        # then skeletonizes. Background color is auto-detected from (0,0).
        # Asserts that the output is not all white.
        # Saves the output bitmap to the temp folder.
        image_path = resources_dir / "silhouette_man_running.png"

        # Read the image in grayscale
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to read image file"

        # Invert the colors (white background becomes black, black silhouette becomes white)
        inverted_image = 255 - image

        # Skeletonize the inverted image (background color auto-detected from (0,0))
        result = thinning_instance.thin(inverted_image)

        # Assert output is a numpy array
        assert isinstance(result, np.ndarray)

        # Assert output is 2D
        assert result.ndim == 2

        # Assert output is not all white (contains some non-255 values)
        assert not np.all(result == 255), "Output should not be all white"

        # Assert output contains some black pixels (0 values)
        assert np.any(result == 0), "Output should contain black pixels"

        # Save output bitmap to temp folder
        output_path = temp_dir / "silhouette_man_running_inverted_colors_output.png"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), "Output bitmap was not saved"

    def test_skeletonize_with_fattened_size_offset_3(self, resources_dir, temp_dir):
        # Test skeletonization with fattened_size_offset set to 3.
        # Creates a Config with fattened_size_offset=3 to make the skeleton thicker.
        # Loads a PNG file and skeletonizes it. Asserts that the output
        # is not completely white (i.e., contains some black pixels).
        # Saves the output bitmap to the temp folder for visual inspection.
        # Create a Config with fattened_size_offset set to 3
        config = Config(hardware_accelerated=False, fattened_size_offset=3)
        thinning_fattened = Thinning(config)

        image_path = resources_dir / "silhouette_man_running.png"

        # Load image as numpy array
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        assert image is not None, "Failed to read image file"

        # Skeletonize the image with fattening
        result = thinning_fattened.thin(image)

        # Assert output is a numpy array
        assert isinstance(result, np.ndarray)

        # Assert output is 2D
        assert result.ndim == 2

        # Assert output is not all white (contains some non-255 values)
        assert not np.all(result == 255), "Output should not be all white"

        # Assert output contains some black pixels (0 values)
        assert np.any(result == 0), "Output should contain black pixels"

        # Save output bitmap to temp folder
        output_path = temp_dir / "silhouette_man_running_fattened_size_3_output.png"
        cv2.imwrite(str(output_path), result)
        assert output_path.exists(), "Output bitmap was not saved"
