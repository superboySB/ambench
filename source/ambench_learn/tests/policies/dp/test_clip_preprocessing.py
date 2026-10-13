# Copyright (c) 2026, The AM-Bench Contributors.
#
# SPDX-License-Identifier: Apache-2.0

"""Verify pretrained RGB input semantics and deterministic held-out inference."""

import pytest
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as functional
from diffusion_policy.model.vision import timm_obs_encoder

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


class CaptureVisionBackbone(torch.nn.Module):
    """Observe the tensor arriving at an otherwise unrelated visual model."""

    pretrained_cfg = {"mean": CLIP_MEAN, "std": CLIP_STD}

    def forward(self, image):
        self.last_input = image.detach().clone()
        feature = image.mean(dim=(-1, -2))
        return torch.stack((feature, feature * 2), dim=1)


def make_encoder(monkeypatch, **kwargs):
    monkeypatch.setattr(timm_obs_encoder.timm, "create_model", lambda **_: CaptureVisionBackbone())
    return timm_obs_encoder.TimmObsEncoder(
        shape_meta={"obs": {"camera0_rgb": {"type": "rgb", "shape": [3, 224, 224], "horizon": 2}}},
        model_name="vit_base_patch16_clip_224.openai",
        pretrained=True,
        frozen=False,
        global_pool="",
        transforms=[transforms.RandomCrop(212), transforms.Resize(224), transforms.ColorJitter(0.3, 0.4, 0.5, 0.08)],
        feature_aggregation=None,
        **kwargs,
    )


def test_eval_uses_pretrained_statistics_and_a_fixed_field_of_view(monkeypatch):
    encoder = make_encoder(monkeypatch, normalize_rgb=True, eval_crop_ratio=0.95).eval()
    torch.manual_seed(42)
    image = torch.rand(1, 2, 3, 224, 224)
    first = encoder({"camera0_rgb": image})
    arrived = encoder.key_model_map["camera0_rgb"].last_input
    expected = functional.center_crop(image.flatten(0, 1), [212, 212])
    expected = functional.resize(expected, [224, 224], antialias=True)
    expected = functional.normalize(expected, CLIP_MEAN, CLIP_STD)
    torch.testing.assert_close(arrived, expected, rtol=0, atol=0)
    torch.manual_seed(345)
    torch.testing.assert_close(encoder({"camera0_rgb": image}), first, rtol=0, atol=0)
    assert first.shape == (1, 6)
    assert "_rgb_mean" not in encoder.state_dict()
    assert "_rgb_std" not in encoder.state_dict()


def test_training_keeps_augmentation_and_legacy_eval_is_preserved(monkeypatch):
    image = torch.rand(1, 2, 3, 224, 224)
    for encoder in (
        make_encoder(monkeypatch, normalize_rgb=True, eval_crop_ratio=0.95).train(),
        make_encoder(monkeypatch).eval(),
    ):
        torch.manual_seed(42)
        first = encoder({"camera0_rgb": image})
        torch.manual_seed(43)
        second = encoder({"camera0_rgb": image})
        assert not torch.equal(first, second)
    legacy_input = encoder.key_model_map["camera0_rgb"].last_input
    assert legacy_input.min() >= 0 and legacy_input.max() <= 1


@pytest.mark.parametrize("ratio", [0, -0.5, 1.01])
def test_invalid_eval_crop_is_rejected(monkeypatch, ratio):
    with pytest.raises(ValueError, match="eval_crop_ratio"):
        make_encoder(monkeypatch, eval_crop_ratio=ratio)
