import pytest
import numpy as np

from ovizioapi.reconstruct import (
    reconstruct_phase,
    reconstruct_intensity,
    reconstruct_hologram,
)
from ovizioapi.metadata import get_metadata


def test_reconstruct_phase(demo_capture_path, demo_phase_paths):
    n = get_metadata(demo_capture_path)["count"]
    for j in range(n):
        image = reconstruct_phase(demo_capture_path, j)
        assert image.shape == (384, 512)
        assert image.dtype == np.float32
        ref = np.load(demo_phase_paths[j])
        assert np.allclose(image, ref, atol=1e-05)


def test_reconstruct_intensity(demo_capture_path, demo_intensity_paths):
    n = get_metadata(demo_capture_path)["count"]
    for j in range(n):
        image = reconstruct_intensity(demo_capture_path, j)
        assert image.shape == (384, 512)
        assert image.dtype == np.uint8
        ref = np.load(demo_intensity_paths[j])
        assert np.array_equal(image, ref)


def test_reconstruct_hologram(demo_capture_path, demo_hologram_paths):
    n = get_metadata(demo_capture_path)["count"]
    for j in range(n):
        image = reconstruct_hologram(demo_capture_path, j)
        assert image.shape == (1536, 2048)
        assert image.dtype == np.uint8
        ref = np.load(demo_hologram_paths[j])
        assert np.array_equal(image, ref)


@pytest.mark.parametrize(
    "reconstruction", [reconstruct_phase, reconstruct_intensity, reconstruct_hologram]
)
def test_reconstruct_file_not_founnd(reconstruction):
    with pytest.raises(FileNotFoundError):
        reconstruction("i/do/not/exist.h5", 0)
