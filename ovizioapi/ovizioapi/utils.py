# Core modules
from copy import deepcopy
from typing import Callable, Tuple

# Third party modules
from deepdiff import DeepDiff
from dask.callbacks import Callback

def do_nothing(*_, **__):
    """This function takes any input arguments and does nothing with them."""
    pass

class CustomCallback(Callback):
    """A custom callback to track the progress of the Dask computation
    which an pipe the output to an arbitrary callback function.

    Compare with TqdmCallback https://tqdm.github.io/docs/dask/
    """

    def __init__(
        self,
        handle: Callable = None,
        desc: str = None,
        start: Callable = None,
        pretask: Callable = None,
    ):
        """Creates a callback object which can track the Dask progress
        and pipe it to a custom callback function.

        If handed over a callback function handle the callback function will be
        in the following manner with five arguments:

        callback(description:str, current:int, total:int, runs:int, finished:bool)

        - description will be the identifier string which was passed at creation
        - current represents the current number of finished tasks
        - total represents the total number of tasks
        - runs counts the number of finish events if there is more then one dask compute
        - finished signalizes that one run is over

        Compare with the default Dask Callback:
        https://docs.dask.org/en/stable/diagnostics-local.html#dask.diagnostics.Callback

        :param handle: callback function handle, defaults to None
        :type handle: Callable, optional
        :param desc: identifier string which will be passed to the callback, defaults to None
        :type desc: str, optional
        :param start: Custom start function, defaults to None
        :type start: Callable, optional
        :param pretask: Custom preteask function, defaults to None
        :type pretask: Callable, optional
        """
        super().__init__(start=start, pretask=pretask)
        self.total = 0
        self.current = 0
        self.runs = 0
        self.callback = handle if handle is not None else do_nothing
        self.description = desc if desc is not None else ""

    def _start_state(self, _, state):
        """Initialize the object before starting"""
        self.total = sum(len(state[k]) for k in ["ready", "waiting", "running", "finished"])

    def _posttask(self, *_, **__):
        """This function will be called everything a task is completed"""
        self.current += 1
        self.callback(self.description, self.current, self.total, self.runs, False)

    def _finish(self, *_, **__):
        """This function will be called if a whole dask.compute graph
        has completed all individual tasks in it.
        """
        self.current = 0
        self.runs += 1
        self.callback(self.description, self.total, self.total, self.runs, True)
