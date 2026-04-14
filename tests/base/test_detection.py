import numpy as np
import pytest
from base.detection import ObjectDetection 
import os


VALID_IMG_PATH = "C:/Users/Admin/Desktop/Projects/dhm/tests/test_data"  # Update this path if necessary
VALID_WEIGHT_PATH = "C:/Users/Admin/Desktop/Projects/dhm/models/28_9_23_yolov8x.pt"  # Update this path if necessary


class TestObjectDetection:

    def setup_config(self):
        return {"detection_batch_size": 2, "img_height": 640, "img_width": 640, "detection_gpu": "cuda:0"}

    def setup_method(self):
        self.config = self.setup_config()
        self.object_detection = ObjectDetection()
        self.object_detection.config = self.config

    def test_normalize_images(self):
        test_img = np.zeros((100, 100), dtype=np.uint8)  # Sample grayscale image
        batch_imgs = [test_img]
        normalized_imgs = self.object_detection.normalize_images(batch_imgs)
        # Assert that the output is as expected
        assert len(normalized_imgs) == 1
        assert normalized_imgs[0].shape == (100, 100, 3)  # Should be a 3-channel image

    def test_extract_index(self):
        # Test extract_index method with different filenames
        assert self.object_detection.extract_index("image123.png") == 123
        assert self.object_detection.extract_index("no_number.png") == -1

    def test_predict_with_real_data(self):
        # Test predict method with actual images and weights
        if not os.path.exists(VALID_IMG_PATH) or not os.path.exists(VALID_WEIGHT_PATH):
            pytest.skip("Valid image path or weight path not found")

        all_images, all_results, image_ids = self.object_detection.predict(VALID_IMG_PATH, VALID_WEIGHT_PATH, [])

        # Assertions to validate the behavior
        assert len(all_images) > 0
        assert len(all_results) == len(all_images)
        assert len(image_ids) == len(all_images)
        assert all(isinstance(image, np.ndarray) for image in all_images)
        assert isinstance(all_images, list)
        assert isinstance(all_results, list)
        assert isinstance(image_ids, list)

    def teardown_method(self):
        pass
