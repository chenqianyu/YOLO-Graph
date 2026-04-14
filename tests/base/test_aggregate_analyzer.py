import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from base.aggregate_analyzer import Aggregate


class MockResult:
    def __init__(self, xywh, cls, conf):
        self.boxes = MagicMock()
        self.boxes.xywh = xywh
        self.boxes.cls = cls
        self.boxes.conf = conf
        
@pytest.fixture
def mock_images_and_results():
    images = [np.random.rand(384, 512) for _ in range(10)]  # Mock images
    img_path = "mock/path"

    results = []
    for _ in range(10):
        # Create mock data for boxes
        xywh = torch.rand(5, 4)  
        cls = torch.randint(0, 3, (5,))  
        conf = torch.rand(5)  
        mock_result = MockResult(xywh, cls, conf)
        results.append(mock_result)

    image_ids = [f"image_{i}" for i in range(10)]
    return images, img_path, results, image_ids

@pytest.fixture
def aggregate_instance(mock_images_and_results):
    images, img_path, results, image_ids = mock_images_and_results
    return Aggregate(images, img_path, results, image_ids)

def test_calculate_all_predictions(aggregate_instance):
    # Test the calculate_all_predictions method
    aspect_ratio_threshold = 0.5
    border_tolerance = 15
    filtered_predictions = aggregate_instance.calculate_all_predictions(
        aggregate_instance.results, aggregate_instance.image_ids,
        aspect_ratio_threshold, border_tolerance
    )
    assert isinstance(filtered_predictions, list)

def test_compute_centroid(aggregate_instance):
    # Create a mock bounding box tensor in the format (x_center, y_center, width, height)
    bbox = torch.tensor([10, 20, 30, 40]).float() 
    # Call the compute_centroid method
    centroid = aggregate_instance.compute_centroid(bbox.unsqueeze(0))  
    assert isinstance(centroid, torch.Tensor)
    assert centroid.shape == (1, 2) 

def test_find_aggregates(aggregate_instance):
    save_predicted_images = True
    aggregates_list, aggregate_image_ids, aggregate_image_info = aggregate_instance.find_aggregates(save_predicted_images)
    assert isinstance(aggregates_list, list)
    assert isinstance(aggregate_image_ids, list)
    assert isinstance(aggregate_image_info, list)
    

