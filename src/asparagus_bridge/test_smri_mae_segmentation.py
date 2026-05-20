import torch

from asparagus.modules.lightning_modules.segmentation_module import SegmentationModule
from asparagus_preprocessing.datasets_segmentation import SEG009_FOMO26_Meningioma_CUSTOM
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


def test_smri_mae_seg_backbone_sliding_window_predict_shape():
    model = _tiny_seg_model(input_channels=1, output_channels=3)
    x = torch.randn(1, 1, 80, 80, 80)

    with torch.no_grad():
        y = model.sliding_window_predict(x, patch_size=(64, 64, 64), overlap=0.5)

    assert y.shape == (1, 3, 80, 80, 80)


def test_smri_mae_seg_backbone_loads_one_channel_encoder_checkpoint():
    source_model = _tiny_seg_model(input_channels=1, output_channels=2)
    target_model = _tiny_seg_model(input_channels=1, output_channels=2)
    source_stem = source_model.encoder.patch_embed.weight.detach().clone()
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

    assert torch.allclose(module.model.encoder.patch_embed.weight, source_stem)


def test_segmentation_module_pads_and_crops_shallow_inference_volume():
    module = SegmentationModule(
        model=_tiny_seg_model(input_channels=1, output_channels=3),
        inference_patch_size=[64, 64, 64],
        compile_mode=None,
    )
    x = torch.randn(1, 1, 64, 64, 21)

    padded, spatial_shape, pad_widths = module._pad_image_for_sliding_window_inference(
        x=x,
        patch_size=module.inference_patch_size,
    )
    logits = torch.randn(1, 3, *padded.shape[2:])
    cropped = module._crop_logits_to_spatial_shape(logits, spatial_shape, pad_widths)

    assert padded.shape == (1, 1, 64, 64, 64)
    assert spatial_shape == (64, 64, 21)
    assert pad_widths == [(0, 0), (0, 0), (21, 22)]
    assert cropped.shape == (1, 3, 64, 64, 21)


def test_segmentation_module_keeps_non_shallow_inference_volume_geometry():
    module = SegmentationModule(
        model=_tiny_seg_model(input_channels=1, output_channels=3),
        inference_patch_size=[64, 64, 64],
        compile_mode=None,
    )
    x = torch.randn(1, 1, 80, 80, 80)

    padded, spatial_shape, pad_widths = module._pad_image_for_sliding_window_inference(
        x=x,
        patch_size=module.inference_patch_size,
    )
    logits = torch.randn(1, 3, *padded.shape[2:])
    cropped = module._crop_logits_to_spatial_shape(logits, spatial_shape, pad_widths)

    assert padded.shape == x.shape
    assert spatial_shape == (80, 80, 80)
    assert pad_widths == [(0, 0), (0, 0), (0, 0)]
    assert cropped.shape == (1, 3, 80, 80, 80)


def test_segmentation_module_test_step_handles_shallow_inference_volume():
    module = SegmentationModule(
        model=_tiny_seg_model(input_channels=1, output_channels=3),
        inference_patch_size=[64, 64, 64],
        compile_mode=None,
    )
    module.on_test_epoch_start()
    batch = {
        "image": torch.randn(1, 1, 64, 64, 21),
        "properties": {
            "pad_box": [],
            "crop_box": [],
            "original_size": (64, 64, 21),
            "size_before_resample": (64, 64, 21),
        },
        "src_label": torch.zeros(1, 1, 64, 64, 21, dtype=torch.long),
        "file_path": "task2_shallow.pt",
    }

    with torch.no_grad():
        module.test_step(batch, batch_idx=0)

    assert "task2_shallow.pt" in module.results


def test_task2_custom_segmentation_default_modalities_are_flair():
    assert SEG009_FOMO26_Meningioma_CUSTOM.normalize_modalities() == ["flair"]


def test_task2_custom_segmentation_accepts_dwi_modality():
    assert SEG009_FOMO26_Meningioma_CUSTOM.normalize_modalities("dwi") == ["dwi"]


def test_task2_custom_segmentation_rejects_unknown_modality():
    try:
        SEG009_FOMO26_Meningioma_CUSTOM.normalize_modalities(["t1"])
    except ValueError as exc:
        assert "Unknown Task 2 modalities" in str(exc)
    else:
        raise AssertionError("Expected unknown Task 2 modality to raise ValueError")


def test_task2_custom_segmentation_builds_modality_metadata(monkeypatch, tmp_path):
    captured = {}

    def fake_postprocess(**kwargs):
        captured["dataset_config"] = kwargs["dataset_config"]

    monkeypatch.setattr(SEG009_FOMO26_Meningioma_CUSTOM, "get_data_path", lambda: str(tmp_path))
    monkeypatch.setattr(
        SEG009_FOMO26_Meningioma_CUSTOM,
        "simple_recursive_find_and_group_files",
        lambda *args, **kwargs: ([], []),
    )
    monkeypatch.setattr(
        SEG009_FOMO26_Meningioma_CUSTOM,
        "process_dataset_without_table",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        SEG009_FOMO26_Meningioma_CUSTOM,
        "simple_postprocess_standard_dataset",
        fake_postprocess,
    )

    SEG009_FOMO26_Meningioma_CUSTOM.main(
        path=str(tmp_path),
        processes=1,
        save_as_tensor=True,
    )

    dataset_config = captured["dataset_config"]
    assert dataset_config.task_name == "SEG009_FOMO26_Meningioma_FLAIR"
    assert dataset_config.n_modalities == 1
    assert dataset_config.modalities == ["flair"]
    assert dataset_config.channel_names == {"0": "flair"}
