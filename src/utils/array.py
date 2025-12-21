# src/utils/array.py
import numpy as np

def clip_u(u, eps: float = 1e-10):
    u = np.asarray(u, dtype=float)
    return np.clip(u, eps, 1.0 - eps)
