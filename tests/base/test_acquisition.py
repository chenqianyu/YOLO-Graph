
import unittest
import numpy as np
import pytest
import os
from unittest.mock import MagicMock
from base.acquisition import Acquisition


VALID_H5_PATH = "C:/Users/Admin/Desktop/Projects/dhm/tests/test_data/test.h5"
INPUT_PATH = "C:/Users/Admin/Desktop/Projects/dhm/tests/test_data"

class TestAcquisition(unittest.TestCase):
    
    @pytest.fixture(autouse=True)
    def setup_acquisition(self):
        app = MagicMock()
        config = {"detection_batch_size": 2, "chunk_size": 2, "img_height": 640, "ovizio_detection_model": "C:/Users/Admin/Desktop/Projects/dhm/models/28_9_23_yolov8x.pt",
                "img_width": 640, "detection_gpu": "cuda:0", "reconstruction_gpu": "cuda:0", "save_predicted_images": False, "containerization": False,
                "img_output": "/path_img",}
        self.acquisition = Acquisition(app, config)

    def test_get_images_slice(self):
        self.acquisition.holo_images = [np.zeros((100, 100)) for _ in range(10)]
        result = self.acquisition.get_images_slice(2, 5)
        assert len(result) == 3
        assert isinstance(result[0], np.ndarray)

    def test_load_h5file_with_real_data(self):
        # Ensure the h5 file path is valid
        if not os.path.exists(VALID_H5_PATH):
            pytest.skip("Valid .h5 file path not found")
        self.acquisition.load_h5file(VALID_H5_PATH)
        # Assertions to validate the behavior
        assert len(self.acquisition.holo_images) > 0
        assert all(isinstance(image, np.ndarray) for image in self.acquisition.holo_images)
        assert self.acquisition.holo_images[0].shape == (1536, 2048)

    def test_reset_aggregate_counts(self):
        self.acquisition.rbc_count = 10
        self.acquisition.reset_aggregate_counts()
        assert self.acquisition.rbc_count == 0

    def test_reconstruct_images_with_real_data(self):
        self.acquisition.holo_images = [np.random.rand(100, 100) for _ in range(10)]
        success = self.acquisition.reconstruct_images(save_images=False, ovizio_reconstruction=True, h5_file_path=VALID_H5_PATH)
        assert success
        assert len(self.acquisition.phase_images) > 0

    def test_process_data_with_real_data(self):
        self.acquisition.process_data(INPUT_PATH)
        assert self.acquisition.rbc_count == 0
        assert self.acquisition.wbc_count == 0
        assert self.acquisition.plt_count == 0
        assert self.acquisition.plt_plt_count == 0
        assert self.acquisition.wbc_plt_count == 0
        assert self.acquisition.wbc_wbc_count == 0
        assert self.acquisition.aggregate_class_count == []


