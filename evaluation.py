"""
Evaluator for circle packing example (n=26) with improved timeout handling
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
from typing import Tuple, Optional


def validate_packing(
    centers: np.ndarray, radii: np.ndarray, reported_sum: float, atol=1e-6,
) -> Tuple[bool, Optional[str]]:
    """
    Validates circle packing results based on the output of 'run_packing'.

    Args:
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        reported_sum: Reported sum of all radii
        atol: Absolute tolerance for floating point comparisons

    Returns:
        (is_valid: bool, error_message: Optional[str])
    """
    centers, radii, reported_sum
    msg = "The circles are placed correctly. There are no overlaps or any circles outside the unit square."
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)

    n_expected = 26
    if centers.shape != (n_expected, 2):
        msg = (
            f"Centers shape incorrect. Expected ({n_expected}, 2), got {centers.shape}"
        )
        return False, msg
    if radii.shape != (n_expected,):
        msg = f"Radii shape incorrect. Expected ({n_expected},), got {radii.shape}"
        return False, msg

    if not np.all((centers >= 0) & (centers <= 1)):
        msg = "Center outside the unit square."
        return False, msg

    if not np.all((radii >= 0) & (radii <= 1)):
        msg = "Radius outside [0, 1]."
        return False, msg

    if not np.isclose(np.sum(radii), reported_sum, atol=atol):
        msg = (
            f"Sum of radii ({np.sum(radii):.6f}) does not match "
            f"reported ({reported_sum:.6f})"
        )
        return False, msg

    for i in range(n_expected):
        x, y = centers[i]
        r = radii[i]
        is_outside = (
            x - r < -atol or x + r > 1 + atol or y - r < -atol or y + r > 1 + atol
        )
        if is_outside:
            msg = (
                f"Circle {i} (x={x:.4f}, y={y:.4f}, r={r:.4f}) is outside unit square."
            )
            return False, msg

    for i in range(n_expected):
        for j in range(i + 1, n_expected):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - atol:
                msg = (
                    f"Circles {i} & {j} overlap. Dist: {dist:.4f}, "
                    f"Sum Radii: {(radii[i] + radii[j]):.4f}"
                )
                return False, msg
    return True, msg


def _produce(out_path: str) -> None:
    from main import construct_packing

    centers, radii = construct_packing()
    np.savez(
        out_path,
        centers=np.asarray(centers, dtype=float),
        radii=np.asarray(radii, dtype=float),
    )


def run_construct_packing() -> Tuple[np.ndarray, np.ndarray]:
    fd, out_path = tempfile.mkstemp(suffix=".npz")
    os.close(fd)
    try:
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--produce", out_path],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if proc.returncode != 0:
            raise ValueError(
                f"construct_packing subprocess exited with {proc.returncode}"
            )
        with np.load(out_path, allow_pickle=False) as data:
            return data["centers"], data["radii"]
    finally:
        os.unlink(out_path)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--produce":
        _produce(sys.argv[2])
        sys.exit(0)

    # Run circle packing
    centers, radii = run_construct_packing()
    sum_radii = np.sum(radii)

    # Validate the packing
    is_valid, error_message = validate_packing(centers, radii, sum_radii)
    if not is_valid:
        raise ValueError(f"Validation failed: {error_message}")

    # Output the results
    print(json.dumps({"output": {"fitness": sum_radii, "signature": (sum_radii,)}, "metainfo": "Success"}))
