# core modules
from threading import Thread, RLock
from tqdm.auto import trange, tqdm

# Third party modules
import numpy as np
from cellface.storage.container import Raw, Capture
from cellface.storage.dataset import DataSet
from cellface.storage.group import HologramGroup

# First party modules
from ovizioapi.capture import OvizioCapture

lock = RLock()


class WriterTread(Thread):
    desc = ""

    def __init__(self, source: OvizioCapture, target: DataSet):
        # self.lock = Lock()
        self.source = source
        self.target = target
        self.resize_target()
        Thread.__init__(self)

    @property
    def source_size(self):
        return 2900# len(self.source)

    def retrieve(self, index) -> np.ndarray:
        raise NotImplementedError

    def resize_target(self) -> None:
        self.target.resize(self.source_size, axis=0)

    def write(self, index: int) -> None:
        data = self.retrieve(index)
        with lock:
            self.target[index] = data

    def run(self) -> None:
        for idx in trange(self.source_size, desc=self.desc):
            self.write(idx)


class PhaseWriter(WriterTread):
    desc = "phase    "

    def retrieve(self, index) -> np.ndarray:
        # print(f"phase: {index}")
        return self.source.get_phase(index)


class AmplitudeWriter(WriterTread):
    desc = "amplitude"

    def retrieve(self, index) -> np.ndarray:
        # print(f"amplitude {index}")
        return self.source.get_intensity(index)


class HologramWriter(Thread):
    desc = "hologram "

    def __init__(self, source: OvizioCapture, target: HologramGroup):
        # self.lock = Lock()
        self.source = source
        self.target = target
        self.resize_target()
        Thread.__init__(self)

    @property
    def source_size(self):
        return 2900 #len(self.source)

    def resize_target(self) -> None:
        self.target.images.resize(self.source_size, axis=0)
        self.target.capture_metadata.timestep_metadata.resize(self.source_size, axis=0)

    def write_metadata(self) -> None:
        with lock:
            self.target.load_from_capture(self.source.path, exclude_image=True)

    def write(self, capture: Capture, index: int) -> None:
        # print(f"holo: {index}")
        image, attrs = capture.read_time_step(index)
        with lock:
            self.target.images[index] = image
            for key in self.target.capture_metadata.timestep_metadata.reference.dtype.names:
                self.target.capture_metadata.timestep_metadata[index, key] = attrs[key]

    def run(self) -> None:
        self.write_metadata()
        with Capture(self.source.path, mode="r", eco=True) as capture:
            for idx in trange(self.source_size, desc=self.desc):
                self.write(capture, idx)


if __name__ == "__main__":
    source_capture = "C:\\cpt\\Capture 2.h5"
    dest_raw = "C:\\cpt\\Capture 2.raw"

    cpt = OvizioCapture(source_capture)
    with Raw(dest_raw, mode="w") as raw:
        # Build pure strucutre
        raw.create_structure()
        writers = [
            PhaseWriter(source=cpt, target=raw.content.phase.images),
            #AmplitudeWriter(source=cpt, target=raw.content.amplitude.images),
            #HologramWriter(source=cpt, target=raw.content.hologram)
        ]
        for writer in writers:
            writer.start()
        for writer in tqdm(writers, desc='total'):
            writer.join()
