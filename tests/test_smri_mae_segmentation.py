import torch

from asparagus.modules.lightning_modules.segmentation_module import SegmentationModule
from asparagus_bridge.models_smri_mae import SmriMaeSegBackbone


def _tiny_seg_model(input_channels: int, output_channels: int, img_size=(64, 64, 64)):
    return SmriMaeSegBackbone(
        input_channels=input_channels,
        output_channels=output_channels,
        img_size=img_size,
        patch_size=16,
        depth=1,
        embed_dim=24,
        num_heads=4,
    )


def test_smri_mae_seg_backbone_one_channel_shape():
    model = _tiny_seg_model(input_channels=1, output_channels=3)
    x = torch.randn(1, 1, 64, 64, 64)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 3, 64, 64, 64)


def test_smri_mae_seg_backbone_two_channel_shape():
    model = _tiny_seg_model(input_channels=2, output_channels=2)
    x = torch.randn(1, 2, 64, 64, 64)

    with torch.no_grad():
        y = model(x)

    assert y.shape == (1, 2, 64, 64, 64)


def test_smri_mae_seg_backbone_sliding_window_predict_shape():
    model = _tiny_seg_model(input_channels=1, output_channels=3)
    x = torch.randn(1, 1, 80, 80, 80)

    with torch.no_grad():
        y = model.sliding_window_predict(x, patch_size=(64, 64, 64), overlap=0.5)

    assert y.shape == (1, 3, 80, 80, 80)


def test_smri_mae_seg_backbone_repeats_linear_patch_embed_stem_for_multichannel_load():
    source_model = _tiny_seg_model(input_channels=1, output_channels=2)
    target_model = _tiny_seg_model(input_channels=2, output_channels=2)
    source_stem = source_model.encoder.patch_embed.weight.detach().clone()
    expected_stem = source_stem.repeat(1, 2) / 2
    weights = {
        f"model.{name}": value.detach().clone()
        for name, value in source_model.state_dict().items()
        if name.startswith("encoder.")
    }

    module = SegmentationModule(
        model=target_model,
        weights=weights,
        inference_patch_size=[64, 64, 64],
        compile_mode=None,
        load_decoder=False,
        repeat_stem_weights=True,
    )

    assert torch.allclose(module.model.encoder.patch_embed.weight, expected_stem)
