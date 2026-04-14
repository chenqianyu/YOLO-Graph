import os
import numpy as np
from enum import Enum
from ctypes import c_float, c_uint8, POINTER, cast

from OvizioApiNet.Computations import ImageReconstruction
from OvizioApiNet.Computations import ImageReconstructionProfile
from OvizioApiNet.Image import ImageType


class PointerType(Enum):
    FLOAT = c_float, np.float32
    UINT8 = c_uint8, np.uint8


def pointer_to_numpy(pointer, n_pixels: int, ptr_type: PointerType) -> np.ndarray:
    """Transforms the pointer to a image into a numpy accessable array.

    :param pointer: Poiter to the image
    :type pointer: Ovizio Image Pointer
    :param n_pixels: expected number of pixels of the image
    :type n_pixels: int
    :param ptr_type: data type of the image
    :type ptr_type: PointerTypes
    :raises TypeError: [description]
    :return: [description]
    :rtype: np.ndarray
    """
    if ptr_type not in PointerType:
        raise TypeError()

    # ctypes pointer
    c_ptr = ptr_type.value[0]
    # numpy dtype
    dtype = ptr_type.value[1]
    # Cast pointer to iterable
    iterable = cast(pointer.ToInt64(), POINTER(c_ptr))
    # Create numpy vector with n_pixels
    vector = np.fromiter(iterable, dtype=dtype, count=n_pixels)
    return vector


def to_numpy(image, ptr_type=PointerType.FLOAT) -> np.ndarray:
    """[summary]

    :param image: [description]
    :type image: [type]
    :param ptr_type: [description], defaults to PointerTypes.FLOAT
    :type ptr_type: [type], optional
    :return: [description]
    :rtype: np.ndarray
    """
    # Read Pointer to Ovizio Image
    ptr = image.GetPtr()
    # Calcualte the number of pixels
    img_size = image.Width * image.Height
    # Get a numpy readable image vector
    vector = pointer_to_numpy(ptr, img_size, ptr_type)
    # return the reshaped vector as a matrix
    return vector.reshape(image.Height, image.Width)


def reconstruct_phase(path: str, idx: int) -> np.ndarray:
    """Reconstruct the phase image of the selected capture.
    The timeStep indices start at 0.

    :param path: Path to a Capture file
    :type path: str
    :param idx: timeStep index
    :type idx: int
    :raises FileNotFoundError: [description]
    :return: [description]
    :rtype: np.ndarray
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist!")

    image = ImageReconstruction.ReconstructImage(
        path, idx, 0, ImageReconstructionProfile.PhaseProfile, ImageType.Real
    )
    return to_numpy(image, ptr_type=PointerType.FLOAT)


def reconstruct_intensity(path: str, idx: int) -> np.ndarray:
    """Reconstruct the amplitude image of the selected capture.
    The timeStep indices start at 0.

    :param path: Path to a Capture file
    :type path: str
    :param idx: timeStep index
    :type idx: int
    :raises FileNotFoundError: [description]
    :return: [description]
    :rtype: np.ndarray
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist!")

    image = ImageReconstruction.ReconstructImage(
        path, idx, 0, ImageReconstructionProfile.IntensityProfile, ImageType.UChar
    )
    return to_numpy(image, ptr_type=PointerType.UINT8)


def reconstruct_hologram(path: str, idx: int) -> np.ndarray:
    """Reconstruct the hologram image of the selected capture.
    The timeStep indices start at 0.

    :param path: Path to a Capture file
    :type path: str
    :param idx: timeStep index
    :type idx: int
    :raises FileNotFoundError: [description]
    :return: [description]
    :rtype: np.ndarray
    """

    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist!")

    image = ImageReconstruction.ReconstructImage(
        path, idx, 0, ImageReconstructionProfile.HologramProfile, ImageType.UChar
    )
    return to_numpy(image, ptr_type=PointerType.UINT8)
