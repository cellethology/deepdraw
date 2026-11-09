"""Normalized positional six-mer DNA kernel."""

import ctypes
import os
import shlex
import subprocess
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parent
WIDTH = 128


@lru_cache(maxsize=1)
def _native_build():
    directory = TemporaryDirectory(prefix="deepdraw-friedman-")
    lib = Path(directory.name) / "kernel.so"
    compiler = shlex.split(os.environ.get("CXX") or "c++")
    try:
        subprocess.run(
            compiler
            + [
                "-O3",
                "-std=c++17",
                "-shared",
                "-fPIC",
                str(ROOT / "kernel.cpp"),
                "-o",
                str(lib),
            ],
            check=True,
        )
    except FileNotFoundError as error:
        directory.cleanup()
        raise RuntimeError(
            "Friedman requires a C++17 compiler (set CXX or install c++)"
        ) from error
    return lib, directory


def encode(sequence):
    if isinstance(sequence, bytes | np.bytes_):
        sequence = sequence.decode("ascii")
    sequence = sequence.upper()
    if len(sequence) < 6 or set(sequence) - set("ACGT"):
        raise ValueError("Kernel requires at least six unambiguous A/C/G/T bases.")
    lut = np.full(256, 255, dtype=np.uint16)
    lut[np.frombuffer(b"ACGT", dtype=np.uint8)] = np.arange(4)
    bases = lut[np.frombuffer(sequence.encode("ascii"), dtype=np.uint8)]
    out = np.zeros(len(bases) - 5, dtype=np.uint16)
    for p in range(6):
        out = (out << 2) | bases[p : p + len(out)]
    return out


class SequenceKernel:
    def __init__(self, sequences):
        codes = [encode(s) for s in sequences]
        self.n = len(codes)
        if not self.n:
            raise ValueError("Empty sequence population.")
        nt = (max(map(len, codes)) + WIDTH - 1) // WIDTH
        self.ids = np.full((self.n, nt), -1, dtype=np.int32)
        tiles, lookup = [], {}
        for row, seq in enumerate(codes):
            for col, start in enumerate(range(0, len(seq), WIDTH)):
                tile = np.full(WIDTH, 65535, dtype=np.uint16)
                piece = seq[start : start + WIDTH]
                tile[: len(piece)] = piece
                key = tile.tobytes()
                if key not in lookup:
                    lookup[key] = len(tiles)
                    tiles.append(tile)
                self.ids[row, col] = lookup[key]
        self.tiles = np.ascontiguousarray(tiles, dtype=np.uint16)
        self.lib = ctypes.CDLL(str(_native_build()[0]))
        p32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
        p16 = np.ctypeslib.ndpointer(dtype=np.uint16, flags="C_CONTIGUOUS")
        p64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
        self.lib.gs_create.argtypes = [
            p32,
            p16,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        self.lib.gs_create.restype = ctypes.c_void_p
        self.lib.gs_destroy.argtypes = [ctypes.c_void_p]
        self.lib.gs_destroy.restype = None
        self.lib.gs_diag.argtypes = [ctypes.c_void_p, p64]
        self.lib.gs_diag.restype = None
        self.lib.gs_gram.argtypes = [
            ctypes.c_void_p,
            p32,
            ctypes.c_int,
            p32,
            ctypes.c_int,
            p64,
        ]
        self.lib.gs_gram.restype = None
        self.ptr = self.lib.gs_create(self.ids, self.tiles, self.n, nt, WIDTH)
        self.norm = np.empty(self.n, dtype=np.float64)
        self.lib.gs_diag(self.ptr, self.norm)
        if not np.all(np.isfinite(self.norm)) or np.any(self.norm <= 0):
            raise ValueError("Invalid self similarities.")
        self.norm = np.sqrt(self.norm)

    def gram(self, rows, cols):
        if not getattr(self, "ptr", None):
            raise RuntimeError("Sequence kernel is closed")
        rows = np.ascontiguousarray(rows, dtype=np.int32)
        cols = np.ascontiguousarray(cols, dtype=np.int32)
        if rows.ndim != 1 or cols.ndim != 1:
            raise ValueError("Indices must be one-dimensional.")
        if (
            np.any(rows < 0)
            or np.any(cols < 0)
            or np.any(rows >= self.n)
            or np.any(cols >= self.n)
        ):
            raise IndexError("Sequence index out of range.")
        out = np.empty((len(rows), len(cols)), dtype=np.float64)
        self.lib.gs_gram(self.ptr, rows, len(rows), cols, len(cols), out)
        out /= self.norm[rows, None]
        out /= self.norm[None, cols]
        return out

    def close(self):
        if getattr(self, "ptr", None):
            self.lib.gs_destroy(self.ptr)
            self.ptr = None

    def __del__(self):
        self.close()
