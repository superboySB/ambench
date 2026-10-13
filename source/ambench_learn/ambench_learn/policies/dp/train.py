# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Run the packaged UMI Diffusion Policy training workspace."""

from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import OmegaConf

CONFIG_DIRECTORY = Path(__file__).parent / "universal_manipulation_interface/diffusion_policy/config"
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(
    version_base=None,
    config_path=str(CONFIG_DIRECTORY),
    config_name="train_diffusion_unet_timm_umi_workspace",
)
def main(cfg: OmegaConf) -> None:
    """Resolve one configuration and execute its existing workspace."""
    OmegaConf.resolve(cfg)
    workspace_class = hydra.utils.get_class(cfg._target_)
    workspace_class(cfg).run()


if __name__ == "__main__":
    main()
