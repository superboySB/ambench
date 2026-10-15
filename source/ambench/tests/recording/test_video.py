# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU regression checks that MP4 encoding retains the last sampled frame."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from ambench.recording.video import _EpisodeVideoWriter


@pytest.mark.parametrize("frame_count", [30, 256])
def test_mp4_keeps_every_sampled_frame(tmp_path, frame_count: int) -> None:
    writer = _EpisodeVideoWriter(tmp_path, fps=30, frame_skip=4, output_format="mp4", gc_trigger=None)
    for step in range(frame_count * 4):
        value = 240 if step == (frame_count - 1) * 4 else 16
        writer.add_step({"ee_camera": np.full((16, 16, 3), value, dtype=np.uint8)})
    writer.finalize_episode(0)

    capture = cv2.VideoCapture(str(tmp_path / "video-ee_camera-env0-eps0.mp4"))
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FPS) == pytest.approx(30)
    decoded = []
    while True:
        okay, frame = capture.read()
        if not okay:
            break
        decoded.append(frame)
    capture.release()

    assert len(decoded) == frame_count
    assert float(decoded[0].mean()) == pytest.approx(16, abs=3)
    assert float(decoded[-1].mean()) == pytest.approx(240, abs=3)
    assert writer.has_data is False
