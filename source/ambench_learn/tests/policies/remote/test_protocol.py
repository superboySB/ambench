# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Protocol tests that run without Isaac Sim or a model checkpoint."""

from __future__ import annotations

from http.server import HTTPServer
from threading import Thread
from unittest import TestCase

import numpy as np

from ambench_learn.policies.remote.protocol import (
    RemotePolicyClient,
    decode_value,
    dumps,
    loads,
    make_handler,
)


class TestRemoteProtocol(TestCase):
    def test_array_round_trip_preserves_camera_layout_and_action_values(self) -> None:
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        actions = np.array([[1.25, -2.5, 0.0]], dtype=np.float32)
        restored = loads(dumps({"observation": {"ee_camera": image}, "actions": actions}))
        np.testing.assert_array_equal(restored["observation"]["ee_camera"], image)
        np.testing.assert_array_equal(restored["actions"], actions)
        self.assertEqual(restored["actions"].dtype, np.float32)

    def test_array_decoder_rejects_invalid_shape_and_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "byte count"):
            decode_value({"__ndarray__": "AAAAAA==", "dtype": "float32", "shape": [2]})
        with self.assertRaisesRegex(ValueError, "dtype"):
            decode_value({"__ndarray__": "", "dtype": "object", "shape": [0]})

    def test_http_client_carries_reset_observations_and_action_chunks(self) -> None:
        class Backend:
            policy_name = "act"

            def info(self):
                return {"policy": self.policy_name, "image_features": {"observation.images.ee_camera": [3, 4, 5]}}

            def reset(self, *, action_semantics):
                assert action_semantics == "ee_absolute"
                return {
                    "action_representation": "ee_local_relative",
                    "state_keys": ["ee_pos", "ee_quat", "gripper_width"],
                }

            def infer(self, *, observation):
                assert observation["observation.images.ee_camera"].shape == (1, 3, 4, 5)
                np.testing.assert_array_equal(observation["observation.state"], np.ones((1, 8), dtype=np.float32))
                return {"action": np.zeros((1, 8), dtype=np.float32)}

        server = HTTPServer(("127.0.0.1", 0), make_handler(Backend()))
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RemotePolicyClient(f"http://127.0.0.1:{server.server_port}")
            self.assertEqual(client.info()["policy"], "act")
            self.assertEqual(client.reset(action_semantics="ee_absolute")["action_representation"], "ee_local_relative")
            result = client.infer(
                observation={
                    "observation.state": np.ones((1, 8), dtype=np.float32),
                    "observation.images.ee_camera": np.zeros((1, 3, 4, 5), dtype=np.float32),
                }
            )
            self.assertEqual(result["action"].shape, (1, 8))
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)
