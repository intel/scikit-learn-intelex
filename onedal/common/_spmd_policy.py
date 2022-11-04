from onedal import _backend, _is_dpc_backend
import sys

class _SPMDDataParallelInteropPolicy(_backend.spmd_data_parallel_policy):
    def __init__(self, queue):
        self._queue = queue
        super().__init__(self._queue)
