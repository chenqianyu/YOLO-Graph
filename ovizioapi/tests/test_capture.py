import numpy as np
from ovizioapi.capture import OvizioCapture


def test_capture_length(demo_capture_path):
    cpt = OvizioCapture(demo_capture_path)
    assert len(cpt) == 10


def test_get_phase(demo_capture_path):
    cpt = OvizioCapture(demo_capture_path)
    for j in range(len(cpt)):
        image = cpt.get_phase(j)
        assert image.shape == (384, 512)
        assert image.dtype == np.float32


def test_get_intensity(demo_capture_path):
    cpt = OvizioCapture(demo_capture_path)
    for j in range(len(cpt)):
        image = cpt.get_intensity(j)
        assert image.shape == (384, 512)
        assert image.dtype == np.uint8


def test_get_hologram(demo_capture_path):
    cpt = OvizioCapture(demo_capture_path)
    for j in range(len(cpt)):
        image = cpt.get_hologram(j)
        assert image.shape == (1536, 2048)
        assert image.dtype == np.uint8
