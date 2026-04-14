# core modules
from functools import partial
from itertools import product
from multiprocessing import Pool
from typing import Tuple, Callable
from tqdm import tqdm

# Third party modules
import numpy as np
from cellface.storage.container import Raw
from cellface.storage.dataset import DataSet

# First party modules
from ovizioapi.capture import OvizioCapture


class DataSetWriter:
    def __init__(self, source: OvizioCapture, target: str) -> None:
        self.source = source
        self.target = target.split("/")

    @property
    def source_size(self):
        return 100  # len(self.source)

    def retrieve(self, index) -> np.ndarray:
        raise NotImplementedError

    def get_target_dataset(self, container: Raw) -> DataSet:
        node = container.content
        for item in self.target:
            node = getattr(node, item)
        return node

    def resize_target(self, container) -> None:
        dataset = self.get_target_dataset(container)
        dataset.resize(self.source_size, axis=0)

    def _write(self, index, container, array):
        dataset = self.get_target_dataset(container)
        dataset[index] = array

    def write(self, index) -> Tuple[np.ndarray, Callable]:
        return self.retrieve(index), partial(self._write, index)


class PhaseWriter(DataSetWriter):
    def retrieve(self, index) -> np.ndarray:
        return self.source.get_phase(index)


class AmplitudeWriter(DataSetWriter):
    def retrieve(self, index) -> np.ndarray:
        return self.source.get_intensity(index)


class HologramWriter:
    def retrieve(self, index) -> np.ndarray:
        return self.source.get_hologram(index)

    def _write(self, index, container, array):
        dataset = self.get_target_dataset(container)
        dataset[index] = array

    def write(self, index) -> Tuple[np.ndarray, Callable]:
        return self.retrieve(index), partial(self._write, index)

def mix(params):
    idx, writer = params
    return writer.write(idx)


if __name__ == "__main__":
    source_capture = "C:\\cpt\\Capture 2.h5"
    dest_raw = "C:\\cpt\\Capture 2.raw"

    cpt = OvizioCapture(source_capture)

    with Raw(dest_raw, mode="w") as raw, Pool(2) as pool:
        # Build pure strucutre
        raw.create_structure()
        # Copy pure hologram meta data
        raw.content.hologram.load_from_capture(source_capture, exclude_image=True, verbose=True)
        # Default Meta Data from Capture
        raw.content.metadata.load_attributes_from_capture(source_capture)
        # Setup parallel writers
        pha_writer = PhaseWriter(cpt, "phase/images")
        amp_writer = AmplitudeWriter(cpt, "amplitude/images")
        pha_writer.resize_target(raw)
        amp_writer.resize_target(raw)

        n_images = 100  # len(cpt)
        writers = [pha_writer, amp_writer]
        jobs = pool.imap_unordered(mix, product(range(n_images), writers))

        for idx, (value, partial_write) in enumerate(tqdm(jobs)):
            partial_write(raw, value)

        pool.close()
        pool.join()
