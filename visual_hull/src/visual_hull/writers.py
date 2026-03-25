from __future__ import annotations

from pathlib import Path
from typing import Literal

import h5py
import numpy as np
from scipy.io import savemat

from .models import FullReconstructionResult

ExportFormat = Literal["auto", "mat", "h5", "hdf5"]


def _normalize_format(output_path: Path, export_format: ExportFormat) -> str:
    if export_format != "auto":
        return "h5" if export_format == "hdf5" else export_format

    suffix = output_path.suffix.lower()
    if suffix == ".mat":
        return "mat"
    if suffix in {".h5", ".hdf5"}:
        return "h5"
    raise ValueError("Could not infer export format from file extension. Use .mat, .h5, or .hdf5, or pass export_format explicitly.")


def write_reconstruction(
    result: FullReconstructionResult,
    output_path: str | Path,
    *,
    export_format: ExportFormat = "auto",
    compression: str | None = "gzip",
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized = _normalize_format(destination, export_format)

    if normalized == "mat":
        _write_mat(result, destination)
    elif normalized == "h5":
        _write_hdf5(result, destination, compression=compression)
    else:
        raise ValueError(f"Unsupported export format: {normalized}")

    return destination


def _write_mat(result: FullReconstructionResult, output_path: Path) -> None:
    savemat(output_path, result.to_matlab_payload())


def _write_hdf5(result: FullReconstructionResult, output_path: Path, *, compression: str | None) -> None:
    payload = result.to_hdf5_payload()
    with h5py.File(output_path, "w") as handle:
        handle.attrs["format"] = "visual_hull.reconstruction"
        handle.attrs["version"] = "0.1.0"
        handle.attrs["completed"] = bool(result.completed)
        for key, value in payload.items():
            if isinstance(value, (bool, np.bool_)):
                handle.create_dataset(key, data=np.array([[bool(value)]], dtype=np.bool_))
                continue

            array = np.asarray(value)
            if array.dtype == np.dtype("O"):
                raise TypeError(f"Cannot serialize object dtype dataset {key!r} to HDF5.")

            dataset_compression = compression if array.ndim > 0 and array.size > 1 else None
            handle.create_dataset(key, data=array, compression=dataset_compression)
