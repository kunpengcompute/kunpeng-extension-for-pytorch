# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

"""
CPU-compatibility shim for the communication benchmark.

When DeepSpeed is installed, ``get_accelerator`` transparently forwards to
``deepspeed.accelerator.get_accelerator`` (so ``DS_ACCELERATOR=cpu`` behaves
exactly as before).  When DeepSpeed is NOT available (e.g. a pure
torch.distributed run on Windows/CPU), a minimal CPU accelerator stub is
returned so the benchmark can run without DeepSpeed at all.

The stub also provides wall-clock timing events because GPU events
(``torch.cuda.Event``) do not exist on CPU.
"""

import time

try:
    from deepspeed.accelerator import get_accelerator as _ds_get_accelerator

    HAS_DEEPSPEED = True
except Exception:
    HAS_DEEPSPEED = False
    _ds_get_accelerator = None

try:
    from deepspeed.comm import TorchBackend

    HAS_DEEPSPEED_COMM = True
except Exception:
    HAS_DEEPSPEED_COMM = False
    TorchBackend = None


class _CPUAcceleratorStub:
    """Minimal stand-in for ``deepspeed.accelerator.CPU_Accelerator``.

    Only implements the methods exercised by the communication benchmarks.
    """

    def __init__(self):
        self._name = "cpu"
        self._communication_backend_name = "gloo"

    # Device APIs
    def device_name(self, device_index=None):
        return "cpu"

    def device(self, device_index=None):
        return None

    def set_device(self, device_index):
        return

    def current_device(self):
        return int(self._local_rank() or 0)

    def device_count(self):
        local_size = self._local_size()
        return local_size if local_size > 0 else 1

    def synchronize(self, device_index=None):
        return

    def empty_cache(self):
        return

    # Memory APIs
    def total_memory(self, device_index=None):
        try:
            import psutil

            return float(psutil.virtual_memory().total)
        except ImportError:
            pass
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            m = MEMORYSTATUSEX()
            m.dwLength = ctypes.sizeof(m)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return float(m.ullTotalPhys)
        except Exception:
            return float(8 * 1024**3)  # fallback 8 GiB

    def get_memory_usage(self, device_index=None):
        try:
            import psutil

            return float(psutil.Process().memory_info().rss)
        except Exception:
            return 0.0

    # Communication backend helpers
    def communication_backend_name(self):
        return self._communication_backend_name

    def get_compile_backend(self):
        return "inductor"

    # Misc
    def is_synchronized_device(self):
        return True

    def use_host_timers(self):
        return True

    def on_accelerator(self, tensor):
        return str(tensor.device).startswith("cpu")

    def _local_rank(self):
        return self._getenv_int(["LOCAL_RANK", "MPI_LOCALRANKID"])

    def _local_size(self):
        return self._getenv_int(["LOCAL_SIZE", "MPI_LOCALSIZE"])

    @staticmethod
    def _getenv_int(names):
        import os

        for name in names:
            val = os.environ.get(name)
            if val is not None and val.strip().isdigit():
                return int(val)
        return 0


_ACCEL_STUB = _CPUAcceleratorStub()


def get_accelerator():
    """Return the DeepSpeed accelerator if available, otherwise the CPU stub."""
    if HAS_DEEPSPEED:
        return _ds_get_accelerator()
    return _ACCEL_STUB


class _WallClockEvent:
    """Duck-typed replacement for ``torch.cuda.Event`` using wall-clock time.

    ``record()`` captures ``time.perf_counter`` and ``elapsed_time()`` returns
    the difference in milliseconds, matching the CUDA event contract used by
    the benchmark.
    """

    def __init__(self, enable_timing=False):
        self._t = None

    def record(self):
        self._t = time.perf_counter()

    def wait(self):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        if self._t is None or other._t is None:
            return 0.0
        return (other._t - self._t) * 1e3  # milliseconds


def timing_events(device):
    """Create a (start, end) event pair for the requested device."""
    if device == "cpu":
        return _WallClockEvent(), _WallClockEvent()
    if device == "xpu":
        import torch

        return torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
    import torch

    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def default_device():
    """Pick a sane default device: cuda if present, otherwise cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"
