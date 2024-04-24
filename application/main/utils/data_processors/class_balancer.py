from typing import Any, Tuple

import numpy as np
from numpy.typing import NDArray


def auto_resample(x: NDArray[Any], y: NDArray[Any], upsample: bool = True) -> Tuple[NDArray[Any], NDArray[Any]]:
    """
    Simple up-downsampling method
    Args:
        x (NDArray[Any]): Data
        y (NDArray[Any]): Label
        upsample (bool): True if up-sampling, false if down-sampling
    Returns:
        (x, y) (Tuple[NDArray[Any], NDArray[Any]]): tuple of resampled data
    """

    # Identify unique classes and their frequencies
    classes, counts = np.unique(y, return_counts=True)
    if upsample:
        target_count = np.max(counts)
    else:
        target_count = np.min(counts)

    for _class, count in zip(classes, counts):
        class_mask = (y == _class)
        if count == target_count:
            continue
        elif upsample and count < target_count:
            diff = target_count - count
            additional_indices = np.random.choice(np.where(class_mask)[0], size=diff, replace=True)
            x = np.vstack((x, x[additional_indices]))
            y = np.concatenate((y, y[additional_indices]))
        elif not upsample and count > target_count:
            diff = count - target_count
            additional_indices = np.random.choice(np.where(class_mask)[0], size=diff, replace=True)
            x = np.delete(x, additional_indices)
            y = np.delete(y, additional_indices)
    return x, y
