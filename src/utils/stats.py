"""Statistical helpers for preparation summaries.

Extracted in v1.3 from prepare_hdfs.py per convergent reviewer recommendation
(R1 F3 + R2 F8): both prep modules need this; future §1.6 token-length gate
will too. Shared utility avoids cross-module imports between sibling prep
modules.
"""

from __future__ import annotations

import numpy as np

from src.types import LengthDistribution


def compute_length_distribution(values: list[int]) -> LengthDistribution:
    """Percentile summary of a list of integer lengths.

    Returns all zeros when the input is empty (guards against np.percentile
    on an empty array, which would warn and return nan).
    """
    if not values:
        return LengthDistribution(min=0, max=0, p50=0.0, p80=0.0, p95=0.0, p99=0.0)
    arr = np.asarray(values)
    return LengthDistribution(
        min=int(arr.min()),
        max=int(arr.max()),
        p50=float(np.percentile(arr, 50)),
        p80=float(np.percentile(arr, 80)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )
