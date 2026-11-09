"""Feature-only candidate pools and revealed fluorescence measurements."""

import hashlib
import json
from dataclasses import dataclass

import numpy as np

CATEGORIES = (
    "syn_promoter",
    "syn_terminator",
    "syn_spacer",
    "idp",
    "zf",
    "core_promoter",
    "ad",
    "orientation",
    "reporter_spacer",
)
FEATURE_COLUMNS = ("circuit_id", *CATEGORIES, "binding_sites")


@dataclass(frozen=True)
class CandidatePool:
    ids: tuple[str, ...]
    labels: dict[str, tuple[str, ...]]
    codes: dict[str, np.ndarray]
    binding_sites: np.ndarray
    rows: tuple[dict[str, str], ...]

    @classmethod
    def from_rows(cls, rows):
        if not rows or any(set(row) != set(FEATURE_COLUMNS) for row in rows):
            raise ValueError("Candidate rows must contain exactly FEATURE_COLUMNS")
        rows = [{key: str(value).strip() for key, value in row.items()} for row in rows]
        if any(not value for row in rows for value in row.values()):
            raise ValueError("Empty feature value")
        ids = tuple(row["circuit_id"] for row in rows)
        if len(set(ids)) != len(ids):
            raise ValueError("Circuit IDs must be unique")
        sites = np.asarray([float(row["binding_sites"]) for row in rows])
        if (
            not np.isfinite(sites).all()
            or np.any(sites < 1)
            or np.any(sites > 12)
            or np.any(sites != np.floor(sites))
        ):
            raise ValueError("Binding-site counts must be integers from 1 to 12")
        labels = {
            category: tuple(sorted({row[category] for row in rows}))
            for category in CATEGORIES
        }
        codes = {}
        for category in CATEGORIES:
            mapping = {value: index for index, value in enumerate(labels[category])}
            codes[category] = np.asarray(
                [mapping[row[category]] for row in rows], dtype=np.int64
            )
            codes[category].setflags(write=False)
        sites = sites.astype(np.int64)
        sites.setflags(write=False)
        return cls(ids, labels, codes, sites, tuple(rows))

    def fingerprint(self):
        value = json.dumps(self.rows, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(value.encode()).hexdigest()


@dataclass(frozen=True)
class Observations:
    indices: np.ndarray
    log_fluorescence: np.ndarray

    def validate(self, pool):
        indices = np.asarray(self.indices)
        values = np.asarray(self.log_fluorescence)
        if indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu":
            raise ValueError("Observed indices must be a nonempty integer vector")
        if (
            np.any(indices < 0)
            or np.any(indices >= len(pool.ids))
            or len(np.unique(indices)) != len(indices)
        ):
            raise ValueError("Invalid or repeated observation indices")
        if values.shape != (len(indices), 2) or not np.isfinite(values).all():
            raise ValueError("Expected finite log basal/induced fluorescence pairs")
        return self

    def fingerprint(self):
        order = np.argsort(self.indices)
        payload = np.asarray(self.indices[order], dtype="<i8").tobytes()
        payload += np.asarray(self.log_fluorescence[order], dtype="<f8").tobytes()
        return hashlib.sha256(payload).hexdigest()
