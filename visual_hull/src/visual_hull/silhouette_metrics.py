from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.ndimage import binary_fill_holes
from skimage.draw import polygon

from .camera import OpenLPTCameraSet


Mesh = tuple[np.ndarray, np.ndarray]


def render_meshes_to_mask(
    meshes: Sequence[Mesh],
    cameras: OpenLPTCameraSet,
    camera_index: int,
    image_shape: tuple[int, int],
) -> np.ndarray:
    rendered = np.zeros(image_shape, dtype=bool)

    for vertices, faces in meshes:
        vertex_array = np.asarray(vertices, dtype=np.float64)
        face_array = np.asarray(faces, dtype=np.int64)
        if vertex_array.size == 0 or face_array.size == 0:
            continue

        projection = cameras.project_points(camera_index, vertex_array)
        for face in face_array:
            if not np.all(projection.valid[face]):
                continue
            rows, cols = polygon(
                projection.pixels[face, 1],
                projection.pixels[face, 0],
                shape=image_shape,
            )
            rendered[rows, cols] = True

    return np.asarray(binary_fill_holes(rendered), dtype=bool)


def project_meshes_to_camera_masks(
    meshes: Sequence[Mesh],
    masks: Sequence[np.ndarray],
    cameras: OpenLPTCameraSet,
) -> list[np.ndarray]:
    predicted_masks: list[np.ndarray] = []
    for camera_index, mask in enumerate(masks):
        image_shape = tuple(int(value) for value in np.asarray(mask).shape[:2])
        predicted_masks.append(render_meshes_to_mask(meshes, cameras, camera_index, image_shape))
    return predicted_masks


def _safe_ratio(numerator: int, denominator: int, empty_value: float) -> float:
    if denominator == 0:
        return float(empty_value)
    return float(numerator) / float(denominator)


def mask_overlap_metrics(predicted_mask: np.ndarray, reference_mask: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted_mask, dtype=bool)
    reference = np.asarray(reference_mask, dtype=bool)
    if predicted.shape != reference.shape:
        raise ValueError("predicted_mask and reference_mask must have the same shape.")

    true_positive = int(np.count_nonzero(predicted & reference))
    false_positive = int(np.count_nonzero(predicted & ~reference))
    false_negative = int(np.count_nonzero(~predicted & reference))
    true_negative = int(np.count_nonzero(~predicted & ~reference))

    predicted_positive = true_positive + false_positive
    reference_positive = true_positive + false_negative
    union = true_positive + false_positive + false_negative
    total = true_positive + false_positive + false_negative + true_negative

    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "predicted_positive": predicted_positive,
        "reference_positive": reference_positive,
        "iou": _safe_ratio(true_positive, union, 1.0),
        "dice": _safe_ratio(2 * true_positive, predicted_positive + reference_positive, 1.0),
        "precision": _safe_ratio(true_positive, predicted_positive, 1.0 if reference_positive == 0 else 0.0),
        "recall": _safe_ratio(true_positive, reference_positive, 1.0),
        "pixel_accuracy": _safe_ratio(true_positive + true_negative, total, 1.0),
    }


def summarize_mask_overlap(
    predicted_masks: Sequence[np.ndarray],
    reference_masks: Sequence[np.ndarray],
) -> dict[str, object]:
    if len(predicted_masks) != len(reference_masks):
        raise ValueError("predicted_masks and reference_masks must contain the same number of cameras.")

    per_camera: list[dict[str, float | int]] = []
    totals = {
        "true_positive": 0,
        "false_positive": 0,
        "false_negative": 0,
        "true_negative": 0,
    }

    for camera_index, (predicted_mask, reference_mask) in enumerate(zip(predicted_masks, reference_masks), start=1):
        metrics = mask_overlap_metrics(predicted_mask, reference_mask)
        metrics["camera_index"] = camera_index
        per_camera.append(metrics)
        for key in totals:
            totals[key] += int(metrics[key])

    predicted_positive = totals["true_positive"] + totals["false_positive"]
    reference_positive = totals["true_positive"] + totals["false_negative"]
    union = totals["true_positive"] + totals["false_positive"] + totals["false_negative"]
    total = union + totals["true_negative"]
    overall = {
        "camera_count": len(per_camera),
        "true_positive": totals["true_positive"],
        "false_positive": totals["false_positive"],
        "false_negative": totals["false_negative"],
        "true_negative": totals["true_negative"],
        "predicted_positive": predicted_positive,
        "reference_positive": reference_positive,
        "iou": _safe_ratio(totals["true_positive"], union, 1.0),
        "dice": _safe_ratio(2 * totals["true_positive"], predicted_positive + reference_positive, 1.0),
        "precision": _safe_ratio(
            totals["true_positive"],
            predicted_positive,
            1.0 if reference_positive == 0 else 0.0,
        ),
        "recall": _safe_ratio(totals["true_positive"], reference_positive, 1.0),
        "pixel_accuracy": _safe_ratio(totals["true_positive"] + totals["true_negative"], total, 1.0),
    }
    return {
        "overall": overall,
        "per_camera": per_camera,
    }