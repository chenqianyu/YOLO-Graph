import os
import re
from datetime import datetime
import pytz

from OvizioCoreWrapper import HDF5Document


def get_file_handle(path: str) -> HDF5Document:
    """Opens a capture file and returns a file handle.

    :param path: Path to the Capture file
    :type path: str
    :raises FileNotFoundError: Path is invalid
    :raises SystemError: There was another error opening the Capture
    :return: File handle to the capture
    :rtype: HDF5Document
    """
    # Check if the file path exists
    if not os.path.isfile(path):
        raise FileNotFoundError(f"File {path} does not exist!")
    # Create and load the capture object
    h = HDF5Document()
    err = h.Load(str(path))
    str(err)
    # Check if there was an error loading the capture file

    if err == 0 :
        raise SystemError(f"Capture could ne be loaded. Error Code: {err}")
    return h


def get_capture_number(path: str) -> int:
    """Read out the internal capture name

    :param path: Path to the Capture file
    :type path: str
    :return: Number of the Capture
    :rtype: int
    """
    h = get_file_handle(path)
    name = h.GetCaptureName()
    regex = r"(\d+)"
    res = re.search(regex, name)
    cpt_nr = int(res.group()) if res else 0
    return cpt_nr


def get_creation_date(path: str, timezone: str = None) -> datetime:
    """Read out the creation date of the capture. You can specify a timezone
    in which the date will be converted. If no timezone is stated the date will
    be in UTC.

    :param path: Path to the Capture file
    :type path: str
    :param timezone: Target timezone, defaults to None
    :type timezone: str, optional
    :return: Creation date of the Capture
    :rtype: datetime
    """
    h = get_file_handle(path)
    date_time_obj = h.GetCreationDate()
    # Get the UTC string
    utc_string = date_time_obj.ToUniversalTime().ToString()
    # Convert to datetime object
    utc = datetime.strptime(utc_string, r"%d.%m.%Y %H:%M:%S")
    # Set time zone
    date = utc.replace(tzinfo=pytz.timezone("UTC"))
    # Convert to local time
    if timezone is not None:
        new_zone = pytz.timezone(timezone)
        date = utc.astimezone(new_zone)
    # Return result
    return date


def get_number_of_images(path: str) -> int:
    """Read out the number of images contained in the Capture.

    :param path: Path to the Capture file
    :type path: str
    :return: number of images
    :rtype: int
    """
    h = get_file_handle(path)
    n_images = h.GetSequenceLength()
    return n_images


def get_metadata(path: str) -> dict:
    """Read metadata from a capture file.
    MetaData Dict:
    - height: image hight
    - width: image width
    - count: number of images in the capture
    - pixel_width: physical size of a single pixel
    - wave_length: wave length of the light source
    - magnification: objective actual magnification
    - creation_date: date the capture was recorded
    - name: Name of the capture


    :param path: Path to a Capture file
    :type path: str
    :raises FileNotFoundError: The file path does not exist
    :raises SystemError: There was an error reading the file
    :return: Dictionary with capture metadata
    :rtype: dict
    """
    h = get_file_handle(path)

    # Just a small subset of the available metadata
    metadata = {
        "height": h.GetImageHeight(),
        "width": h.GetImageWidth(),
        "count": h.GetSequenceLength(),
        "pixel_width": h.GetPhysicalPixelWidth(),
        "wave_length": h.GetLightSourceWaveLength(),
        "magnification": h.GetMagnification(),
        "creation_date": h.GetCreationDate(),
        "name": h.GetCaptureName(),
    }

    return metadata
