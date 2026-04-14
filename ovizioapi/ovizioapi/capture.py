# Core Modules
import os
from enum import Enum

# Third party modules
import numpy as np
import dask
from dask.delayed import Delayed
from OvizioApiNet.Computations import ImageReconstruction
from OvizioApiNet.Computations import ImageReconstructionProfile
from OvizioApiNet.Image import ImageType

# First party modules
from ovizioapi.metadata import get_metadata
from ovizioapi.reconstruct import to_numpy, PointerType


class MetaType(Enum):
    hologram = 0
    phase = 1
    intensity = 2


class OvizioCapture:
    def __init__(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File {path} does not exist!")
        self.path = path
        self.metadata = get_metadata(self.path)

    def __len__(self):
        return self.metadata["count"]

    def get_image(self, idx, meta_type):
        if meta_type == MetaType.hologram:
            profile = ImageReconstructionProfile.HologramProfile
            image_type = ImageType.UChar
            ptr_type = PointerType.UINT8

        elif meta_type == MetaType.phase:
            profile = ImageReconstructionProfile.PhaseProfile
            image_type = ImageType.Real
            ptr_type = PointerType.FLOAT

        elif meta_type == MetaType.intensity:
            profile = ImageReconstructionProfile.IntensityProfile
            image_type = ImageType.UChar
            ptr_type = PointerType.UINT8

        else:
            raise TypeError(f"MetaType {meta_type} not found!")

        if idx > len(self):
            raise IndexError(f"Index {idx} out of bounds. Capture size {len(self)}")

        image = ImageReconstruction.ReconstructImage(str(self.path), idx, 0, profile, image_type)
        return to_numpy(image, ptr_type=ptr_type)

    def get_phase(self, idx: int) -> np.ndarray:
        return self.get_image(idx, MetaType.phase)

    def get_intensity(self, idx: int) -> np.ndarray:
        return self.get_image(idx, MetaType.intensity)

    def get_hologram(self, idx: int) -> np.ndarray:
        return self.get_image(idx, MetaType.hologram)


class DaskCapture(OvizioCapture):
    @dask.delayed
    def get_phase(self, idx: int) -> Delayed:
        return self.get_image(idx, MetaType.phase)

    @dask.delayed
    def get_intensity(self, idx: int) -> Delayed:
        return self.get_image(idx, MetaType.intensity)

    @dask.delayed
    def get_hologram(self, idx: int) -> Delayed:
        return self.get_image(idx, MetaType.hologram)
