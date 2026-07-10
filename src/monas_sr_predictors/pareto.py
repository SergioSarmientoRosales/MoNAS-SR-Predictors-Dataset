from __future__ import annotations

import numpy as np
import pandas as pd


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.all(a <= b) and np.any(a < b))


def nondominated_mask(points: np.ndarray) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    for i in range(len(points)):
        for j in range(len(points)):
            if i != j and dominates(points[j], points[i]):
                mask[i] = False
                break
    return mask


def nondominated_frame(frame: pd.DataFrame, objective_values: np.ndarray) -> pd.DataFrame:
    return frame.loc[nondominated_mask(objective_values)].reset_index(drop=True)
