# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Recording utilities for dataset and video capture."""

from __future__ import annotations

from typing import Any

__all__ = ["DatasetRecorder", "RecordVideo"]


def __getattr__(name: str) -> Any:
    """Load dataset writing only when recording needs its LeRobot dependency."""
    if name == "DatasetRecorder":
        from .dataset import DatasetRecorder

        return DatasetRecorder
    if name == "RecordVideo":
        from .video import RecordVideo

        return RecordVideo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
