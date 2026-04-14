from abc import abstractmethod
from typing import Tuple, Union, List

import numpy as np
import dask.array as da
from dask.delayed import Delayed
from dask.array import Array
import dask

from cellface.storage.container import Raw, Capture
from cellface.storage.dataset import DataSet
from cellface.storage.group import HologramGroup

# First party modules
from ovizioapi.capture import DaskCapture


class DataSetWriter:
    """Base Class for configurable DataSet writers

    <--         Lazy Dask Computation Graph        --> < HDF5 >
    |single_image_reconstruction(0)|
    |single_image_reconstruction(1)|
    |single_image_reconstruction(2)| > Array > Store > Dataset
    |...                           |
    |single_image_reconstruction(n)|

    """

    def __init__(self, source: DaskCapture, target: DataSet) -> None:
        self.source = source
        self.target = target
        # Target information
        self.i_shape = (1,) + target.element_shape
        self.i_dtype = target.dtype

    @property
    def source_size(self) -> int:
        """Retrieve the size of the source capture

        :return: number of images in the source capture
        :rtype: int
        """
        return len(self.source)

    @abstractmethod
    def retrieve(self, index: int) -> Array:
        """Set up the future dask Array for a certain image at the given index.
        The array is not eagerly calculated so it can be added to the
        computation graph for batch processing.

        :param index: index of the image to retrieve
        :type index: int
        :raises NotImplementedError: Implement this for the specialized datatype
        :return: Dask Array which will result in a reconstructed image
        :rtype: Array
        """
        raise NotImplementedError

    def resize_target(self) -> None:
        """Shape the target DataSet so it can hold all images from the source
        capture.
        """
        self.target.resize(self.source_size, axis=0)

    def construct_dataset_array(self) -> da.Array:
        """Build a dask Array holding the computation commands for the whole
        target dataset. The store() method of this array can be used to trigger
        the writing of the target dataset inside the dask computation graph.

        :return: Dask Array for the target dataset
        :rtype: Array
        """
        # Build a list with lazy retrievals of the individual images
        lazy_image_list = [self.retrieve(i) for i in range(self.source_size)]
        # Concatenate the single images retrievals to a tensor (NxWxH)
        return da.concatenate(lazy_image_list)

    def store(self, return_stored: bool = False, compute: bool = False) -> Union[Array, Delayed, None]:
        """Creates a dask Array for the target dataset and prepares its storing
        operation.

        :param return_stored: Switch to return the stored result, defaults to False
        :type return_stored: bool, optional
        :param compute: If true compute immediately, defaults to False
        :type compute: bool, optional
        :return: Depending on the switches the stored result or None will be returned.
        :rtype: Union[Array, Delayed, None]
        """
        # Adjust the target dataset so it can hold all images of the source
        self.resize_target()
        # Get the dask array for the future target dataset
        array = self.construct_dataset_array()
        # Return the lazy storage operation if not computed instantly
        return array.store(self.target, compute=compute, return_stored=return_stored)


class PhaseWriter(DataSetWriter):
    """DataSet Writer for phase images.

    This writer calls the get_phase method of the Ovizio Capture.
    """

    def retrieve(self, index: int) -> Array:
        # Get lazy image reconstruction
        image = self.source.get_phase(index)
        # Transform to dask array in the shape (1,W,H)
        return da.from_delayed(image[np.newaxis, :], shape=self.i_shape, dtype=self.i_dtype)


class AmplitudeWriter(DataSetWriter):
    """DataSet Writer for amplitude images.

    This writer calls the get_intensity method of the Ovizio Capture.
    """

    def retrieve(self, index: int) -> Array:
        # Get lazy image reconstruction
        image = self.source.get_intensity(index)
        # Transform to dask array in the shape (1,W,H)
        return da.from_delayed(image[np.newaxis, :], shape=self.i_shape, dtype=self.i_dtype)


class HologramWriter:
    def __init__(self, source: Capture, target: HologramGroup) -> None:
        self.source = source
        self.images = target.images
        self.attrs = target.capture_metadata.timestep_metadata
        # Images Target Information
        self.i_dtype = target.images.dtype
        self.i_shape = (1,) + target.images.element_shape
        # Attributes Target Information
        self.a_dtype = target.capture_metadata.timestep_metadata.dtype
        self.a_shape = (1,)

    @property
    def source_size(self) -> int:
        """Retrieve the size of the source capture

        :return: number of images in the source capture
        :rtype: int
        """
        return len(self.source)

    def resize_target(self) -> None:
        """Shape the target DataSet so it can hold all images from the source
        capture. Also adjust the timestep metadata table according to the number
        of images.
        """
        self.images.resize(self.source_size, axis=0)
        self.attrs.resize(self.source_size, axis=0)

    @dask.delayed
    def _retrieve(self, index: int) -> dict:
        """Wrapper function to delay the retrieval of the hologram.

        This function performs the reading of the hologram image and the according
        attributes. The attributes dictionary will be transformed into a structured
        array for an easier handling later. Also the ordering of the attributes
        according to the table in the target dataset is ensured in that way.

        :param index: index of the image to retrieve
        :type index: int
        :return: A dictionary containing the image and the attributes
        :rtype: dict
        """
        # Collect the hologram and the according attributes
        image, attrs = self.source.read_time_step(index)
        # Transform the attributes to a structured array
        attrs = np.array([tuple(attrs[k] for k in self.a_dtype.names)], dtype=self.a_dtype)
        # Return both items in one dict
        return {"image": image, "attrs": attrs}

    def retrieve(self, index: int) -> Tuple[Array, Array]:
        """Set up the future dask Arrays for a certain image at the given index.
        The first array contains the actual hologram image and the second array
        contains the structures attributes. Both arrays are not eagerly calculated
        so they can be added to the computation graph for batch processing.

        :param index: index of the image to retrieve
        :type index: int
        :return: Image Array and Attributes Array
        :rtype: Tuple[Array, Array]
        """
        # Get lazy image and attributes retrieval
        delayed_dict = self._retrieve(index)
        # Construct an image array from the lazy dict with shape (1,W,H)
        image = da.from_delayed(delayed_dict["image"], shape=self.i_shape, dtype=self.i_dtype)
        # Construct an attributes structured array from the lazy dict with shape (1,)
        attrs = da.from_delayed(delayed_dict["attrs"], shape=self.a_shape, dtype=self.a_dtype)
        # Return both lazy arrays
        return image, attrs

    def construct_dataset_array(self) -> Tuple[Array, Array]:
        """Build a dask Array holding the computation commands for the whole
        target dataset. The store() method of this array can be used to trigger
        the writing of the target dataset inside the dask computation graph.

        :return: Complete Image Array and Attributes Array
        :rtype: Tuple[Array, Array]
        """
        # Build two lists for all lazy retrievals
        lazy_image_list = [None] * self.source_size
        lazy_attrs_list = [None] * self.source_size
        # Populate the lists with the lazy retievals
        for i in range(self.source_size):
            lazy_image_list[i], lazy_attrs_list[i] = self.retrieve(i)
        # Lazily concatenate the lists to a big tensor each
        image_array = da.concatenate(lazy_image_list)
        attrs_array = da.concatenate(lazy_attrs_list)
        # Return the tensors
        return image_array, attrs_array

    def store(self, return_stored: bool = False, compute: bool = False) -> List[Union[Array, Delayed, None]]:
        """Creates two dask Arrays for the target datasets and prepares thier storing
        operation.

        :param return_stored: Switch to return the stored result, defaults to False
        :type return_stored: bool, optional
        :param compute: If true compute immediately, defaults to False
        :type compute: bool, optional
        :return: Depending on the switches the stored result or None will be returned.
        :rtype: List[Union[Array, Delayed, None], Union[Array, Delayed, None]]
        """
        # Adjust the target dataset so it can hold all images of the source
        self.resize_target()
        # Get the dask arrays for the future target dataset
        image_array, attrs_array = self.construct_dataset_array()
        # (lazy) image tensor storing
        image_store = image_array.store(self.images, compute=compute, return_stored=return_stored)
        # (lazy) attribute table storing
        attrs_store = attrs_array.store(self.attrs, compute=compute, return_stored=return_stored)
        # Return the lazy storage operation if not computed instantly
        return [image_store, attrs_store]


def capture_to_raw(
    source: str,
    destination: str,
    metadata: dict,
    phase: bool = True,
    amplitude: bool = True,
    hologram: bool = True,
    cache_stats: bool = True,
    callback: dask.callbacks.Callback = None,
    **callback_kwargs
):
    """Stores a Ovizio Capture file as CellFace RAW Container.

    This function uses several dask computation graphs to convert a
    Ovizio Capture into a CellFace RAW Container while being memory
    and computational efficient.

    The meta data is extracted from the source capture file and
    transferred to the RAW container. You can use the `metadata` dict
    to override the default meta data attributes.

    .. code-block: python
        :caption: metadata
        {
            'capture': 1,
            'donor': 'Unknown',
            'experimenter': 'Christian Klenk',
            'experimenterID': 'ga76quz',
            'notes': 'Demo Measurement',
            'sampleType': 'WB',
            'project': 'Demo',
        }

    Using the three flags `phase`, `amplitude` and `hologram` you can
    select which of the data representations should be included in the
    RAW Container. Normally, all three representations are exported.

    The fourth flag `cache_stats` enables the computation of aggregated
    values about the freshly created datasets. This can be useful for
    a quick quality control of the data.

    To track the progress of the computation you can specify a callback.
    This this should be a subclass of dask.callbacks.Callback since we
    will use dask for the computation. If you have a simple callback
    function from another application you can use the
    ovizioapi.utils.CustomCallback and hand over a function handle in
    the callback_kwargs.

    .. code-block: python
        :caption: Callback Example

        def my_callback(*args, **kwagrs):
            ...

        capture_to_raw(..., callback=CustomCallback, handle=my_callback)

    You can also use the from tqdm.dask.TqdmCallback for printing the
    progress in the commandline or in a jupyter notebook.

    .. code-block: python
        :caption: tqdm
        from tqdm.dask import TqdmCallback
        capture_to_raw(..., callback=TqdmCallback)


    :param source: Path to the source capture file
    :type source: str
    :param destination: Path to the future output file
    :type destination: str
    :param metadata: Dictionary to update the default meta data
    :type metadata: dict
    :param phase: Flag to export phase images, defaults to True
    :type phase: bool, optional
    :param amplitude: Flag to export amplitude images, defaults to True
    :type amplitude: bool, optional
    :param hologram: Flag to export hologram images, defaults to True
    :type hologram: bool, optional
    :param cache_stats: Flag to precalculate statistics, defaults to True
    :type cache_stats: bool, optional
    :param callback: callback class or function, defaults to None
    :type callback: dask.callbacks.Callback, optional
    """

    # Create Lazy Capture object
    cpt = DaskCapture(source)
    # Open Raw Container and CellFace Capture object
    with Raw(destination, mode="w") as raw, Capture(cpt.path, mode="r", eco=True) as capture:
        # Build pure structure
        raw.create_structure()
        # Copy pure hologram meta data
        raw.content.hologram.load_from_capture(source, exclude_image=True)
        # Default Meta Data from Capture
        raw.content.metadata.load_attributes_from_capture(source)
        # Override Meta Data with updated information
        raw.content.metadata.attrs.update(metadata)
        # Export Hologram
        if hologram:
            with callback(desc="hologram", **callback_kwargs):
                # Setup writer
                hol_writer = HologramWriter(capture, raw.content.hologram)
                # Execute computation graph until storage
                dask.compute(*hol_writer.store())
        # Export Phase
        if phase:
            with callback(desc="phase", **callback_kwargs):
                # Setup writer
                pha_writer = PhaseWriter(cpt, raw.content.phase.images)
                # Execute computation graph until storage
                dask.compute(pha_writer.store())
        # Export Amplitude
        if amplitude:
            with callback(desc="amplitude", **callback_kwargs):
                # Setup writer
                amp_writer = AmplitudeWriter(cpt, raw.content.amplitude.images)
                # Execute computation graph until storage
                dask.compute(amp_writer.store())
        # Compute Statistics on the new datasets and cache them
        if cache_stats:
            # Compute the statistics
            with callback(desc="statistics", **callback_kwargs):
                raw.cache_statistics(verbose=False)
