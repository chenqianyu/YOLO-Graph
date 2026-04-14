import pytest


@pytest.fixture
def demo_capture_path():
    return "data/Capture 16.h5"


@pytest.fixture
def demo_phase_paths():
    return ["data/phase/%02d.npy" % i for i in range(10)]


@pytest.fixture
def demo_intensity_paths():
    return ["data/intensity/%02d.npy" % i for i in range(10)]


@pytest.fixture
def demo_hologram_paths():
    return ["data/hologram/%02d.npy" % i for i in range(10)]
