# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify fresh OmniHexa model runs with the shared evidence checks."""

from summarize_uaquad_policies import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            task_id="PressButton-Am-OmniHexa-Abs-PID-Direct-v0",
            seeds=(61, 62),
            robot_name="OmniHexa",
            asset_subdir="tilting",
            run_name="tilting-20261009",
        )
    )
