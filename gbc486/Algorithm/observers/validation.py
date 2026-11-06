"""Numeric contracts shared by observer components."""
import numpy as np


def vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    return result.copy()


def positive(value: float, name: str) -> float:
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def spd(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size} x {size} matrix")
    if not np.allclose(result, result.T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be symmetric")
    np.linalg.cholesky(result)
    return result.copy()
