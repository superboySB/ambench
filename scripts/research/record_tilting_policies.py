# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Record the OmniHexa twelve-trial matrix using the shared CPU policy driver."""

from record_uaquad_policies import main

if __name__ == "__main__":
    raise SystemExit(
        main(
            task_id="PressButton-Am-OmniHexa-Abs-PID-Direct-v0",
            asset_subdir="tilting",
            minimum_free_vram_mib=8192,
            eval_timeout_s=1800,
        )
    )
