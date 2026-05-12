from __future__ import annotations

import pytest
import torch

from asparagus_bridge.convert_checkpoint import convert_checkpoint_file, convert_state_dict


def test_convert_state_dict_maps_encoder_keys_and_drops_decoder_keys():
    source = {
        "encoder.patch_embed.weight": torch.ones(2, 2),
        "module.encoder.blocks.0.norm1.weight": torch.ones(2),
        "_orig_mod.encoder.norm.bias": torch.zeros(2),
        "decoder.head.weight": torch.full((2, 2), 3.0),
        "pred_patchify.patch_dim": torch.tensor(1),
    }

    converted = convert_state_dict(source)

    assert set(converted) == {
        "model.encoder.patch_embed.weight",
        "model.encoder.blocks.0.norm1.weight",
        "model.encoder.norm.bias",
    }
    assert torch.equal(converted["model.encoder.patch_embed.weight"], source["encoder.patch_embed.weight"])


def test_convert_state_dict_preserves_already_lightning_prefixed_encoder_keys():
    source = {
        "model.encoder.patch_embed.bias": torch.ones(2),
        "model.head.weight": torch.ones(1, 2),
    }

    converted = convert_state_dict(source)

    assert set(converted) == {"model.encoder.patch_embed.bias"}


def test_convert_state_dict_rejects_checkpoints_without_encoder_keys():
    with pytest.raises(ValueError, match="No MAE encoder weights"):
        convert_state_dict({"decoder.head.weight": torch.ones(2, 2)})


def test_convert_state_dict_rejects_ambiguous_duplicate_mapping():
    with pytest.raises(ValueError, match="ambiguous encoder weights"):
        convert_state_dict(
            {
                "encoder.patch_embed.weight": torch.ones(2, 2),
                "model.encoder.patch_embed.weight": torch.zeros(2, 2),
            }
        )


def test_convert_checkpoint_file_writes_network_weights_and_args(tmp_path):
    input_path = tmp_path / "native.pth"
    output_path = tmp_path / "asparagus.pth"
    torch.save(
        {
            "model": {
                "encoder.patch_embed.weight": torch.ones(2, 2),
                "decoder.head.weight": torch.zeros(2, 2),
            },
            "args": {"img_size": [208, 240, 208], "patch_size": 8},
            "epoch": 3,
        },
        input_path,
    )

    returned = convert_checkpoint_file(input_path, output_path)
    written = torch.load(output_path, map_location="cpu", weights_only=False)

    assert output_path.exists()
    assert set(written) == {"network_weights", "smri_fm_args"}
    assert set(written["network_weights"]) == {"model.encoder.patch_embed.weight"}
    assert written["smri_fm_args"] == {"img_size": [208, 240, 208], "patch_size": 8}
    assert returned["smri_fm_args"] == written["smri_fm_args"]


def test_convert_checkpoint_file_accepts_state_dict_key(tmp_path):
    input_path = tmp_path / "native_state_dict.pth"
    output_path = tmp_path / "asparagus.pth"
    torch.save({"state_dict": {"encoder.patch_embed.bias": torch.ones(2)}}, input_path)

    convert_checkpoint_file(input_path, output_path)
    written = torch.load(output_path, map_location="cpu", weights_only=False)

    assert set(written["network_weights"]) == {"model.encoder.patch_embed.bias"}
