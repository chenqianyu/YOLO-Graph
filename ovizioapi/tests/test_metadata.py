from ovizioapi.metadata import get_metadata


def test_get_metadata(demo_capture_path):
    reference_dict = {
        "height": 1536,
        "width": 2048,
        "count": 10,
        "pixel_width": 3.45,
        "wave_length": 530.0,
        "magnification": 40.0,
    }
    metadata = get_metadata(demo_capture_path)
    for key in reference_dict:
        assert key in metadata
        assert reference_dict[key] == metadata[key]
