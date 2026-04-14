# core modules
from asyncio import as_completed
from functools import partial
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, Callable
from tqdm.auto import trange, tqdm

# Third party modules
import numpy as np
from cellface.storage.container import Raw, Capture
from cellface.storage.dataset import DataSet
from cellface.storage.group import HologramGroup

# First party modules
from ovizioapi.capture import OvizioCapture


class DataSetWriter:
    def __init__(self, source: OvizioCapture, target: DataSet) -> None:
        self.source = source
        self.target = target

    @property
    def source_size(self):
        return len(self.source)

    def retrieve(self, index) -> np.ndarray:
        raise NotImplementedError

    def get_target_dataset(self, container: Raw) -> DataSet:
        node = container.content
        for item in self.target:
            node = getattr(node, item)
        return node

    def resize_target(self) -> None:
        self.target.resize(self.source_size, axis=0)

    def _write(self, index, array):
        self.target[index] = array

    def write(self, index) -> Tuple[np.ndarray, Callable]:
        return self.retrieve(index), partial(self._write, index)


class PhaseWriter(DataSetWriter):
    def retrieve(self, index) -> np.ndarray:
        return self.source.get_phase(index)


class AmplitudeWriter(DataSetWriter):
    def retrieve(self, index) -> np.ndarray:
        return self.source.get_intensity(index)


class HologramWriter:
    def __init__(self, source: Capture, target: HologramGroup) -> None:
        self.source = source
        self.target = target

    @property
    def source_size(self):
        return len(self.source)

    def resize_target(self) -> None:
        self.target.images.resize(self.source_size, axis=0)
        self.target.capture_metadata.timestep_metadata.resize(self.source_size, axis=0)

    def retrieve(self, index) -> np.ndarray:
        return self.source.read_time_step(index)

    def _write(self, index, values):
        image, attrs = values
        self.target.images[index] = image
        for key in self.target.capture_metadata.timestep_metadata.reference.dtype.names:
            self.target.capture_metadata.timestep_metadata[index, key] = attrs[key]

    def write(self, index) -> Tuple[np.ndarray, Callable]:
        return self.retrieve(index), partial(self._write, index)


def capture_to_raw(source, destination, metadata, cache_stats=True):

    cpt = OvizioCapture(source)

    with ThreadPoolExecutor(max_workers=6) as executor, Raw(destination, mode="w") as raw, Capture(
        cpt.path, mode="r", eco=True
    ) as capture:

        # Build pure strucutre
        raw.create_structure()
        # Copy pure hologram meta data
        raw.content.hologram.load_from_capture(source, exclude_image=True)
        # Default Meta Data from Capture
        raw.content.metadata.load_attributes_from_capture(source)
        # OVerride Meta Data
        raw.content.metadata.attrs.update(metadata)

        n_images = len(cpt)
        pha_writer = PhaseWriter(cpt, raw.content.phase.images)
        amp_writer = AmplitudeWriter(cpt, raw.content.amplitude.images)
        hol_writer = HologramWriter(capture, raw.content.hologram)
        pha_writer.resize_target()
        amp_writer.resize_target()
        hol_writer.resize_target()

        writers = [pha_writer, amp_writer, hol_writer]
        # Start processing
        futures = [executor.submit(w.write, i) for i, w in product(range(n_images), writers)]
        with tqdm(total=len(futures), desc="Images") as pbar:
            for future in as_completed(futures):
                value, partial_write = future.result()
                partial_write(value)
                pbar.update(1)

        if cache_stats:
            # Compute the statistics
            raw.cache_statistics(verbose=True)
