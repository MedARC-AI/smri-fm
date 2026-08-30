"""Reproduce the fixed channel normalization used by Tasks 6 and 7.

Samples 2,000 distinct FOMO300K subjects per modality, extracts Walnut final-layer globally
mean-pooled embeddings, and saves their channel means and standard deviations as a [1024, 2]
float32 array.
"""

import argparse
import io
import json
import random
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import smri_mae.mri_data as mri_data
import smri_mae.utils as smri_utils
from fomo_tune.backbone import load_backbone


CHECKPOINT = "hf://medarc/walnut/checkpoints/walnut-v0-1/vitl/sub-52k/checkpoint-last.pth"
MODALITIES = ("t1w", "t2w", "flair", "t1c", "dwi")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--subjects-per-modality", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7338)
    return parser.parse_args()


def select_scans(data_root: Path, count: int, seed: int) -> list[dict]:
    scans_by_modality_and_subject = defaultdict(lambda: defaultdict(list))
    with (data_root / "candidates_train.jsonl").open() as f:
        for line in f:
            row = json.loads(line)
            if row["modality"] in MODALITIES:
                scans_by_modality_and_subject[row["modality"]][row["subject_unit"]].append(row)

    selected = []
    selected_subjects = set()
    modality_order = sorted(
        MODALITIES, key=lambda modality: len(scans_by_modality_and_subject[modality])
    )
    for modality_index, modality in enumerate(modality_order):
        scans_by_subject = scans_by_modality_and_subject[modality]
        available_subjects = sorted(set(scans_by_subject) - selected_subjects)
        rng = random.Random(seed + modality_index)
        modality_subjects = rng.sample(available_subjects, count)
        for subject in modality_subjects:
            scans = sorted(scans_by_subject[subject], key=lambda row: row["key"])
            selected.append(rng.choice(scans))
        selected_subjects.update(modality_subjects)

    assert len(selected) == len(MODALITIES) * count
    assert len(selected_subjects) == len(selected)
    return sorted(selected, key=lambda row: row["key"])


def load_selected_shard(shard_path: Path, selected_by_key: dict[str, dict]) -> list[dict]:
    samples = []
    with tarfile.open(shard_path) as tar:
        members = {member.name: member for member in tar}
        keys = [
            name.removesuffix(".image_values.npy")
            for name in members
            if name.endswith(".image_values.npy")
            and name.removesuffix(".image_values.npy") in selected_by_key
        ]
        for key in keys:
            values_file = tar.extractfile(members[f"{key}.image_values.npy"])
            mask_file = tar.extractfile(members[f"{key}.img_mask.npy"])
            assert values_file is not None and mask_file is not None
            samples.append(
                {
                    "image_values": np.load(io.BytesIO(values_file.read())),
                    "img_mask": np.load(io.BytesIO(mask_file.read())),
                    "key": key,
                }
            )
    return samples


@torch.inference_mode()
def embed_batch(backbone, samples: list[dict], device: torch.device) -> np.ndarray:
    sparse = mri_data.collate(samples, include_meta=False)
    images, masks = mri_data.densify_sparse_image_batch(
        sparse["image_values"], sparse["img_mask"], (1, 208, 240, 208)
    )
    batch = {
        "image": images.to(device, non_blocking=True),
        "mask": masks.to(device, non_blocking=True),
        "affine": torch.eye(4, device=device).expand(len(samples), -1, -1),
    }
    with torch.autocast("cuda", torch.bfloat16):
        output = backbone(batch)
    patch_embeddings = output["patch_embeds"]
    token_mask = output["token_mask"].unsqueeze(-1)
    embeddings = (patch_embeddings * token_mask).sum(dim=1) / token_mask.sum(dim=1)
    return embeddings.float().cpu().numpy()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    smri_utils.random_seed(args.seed)
    smri_utils.configure_flash_sdpa()
    print(smri_utils.get_sha())
    print(vars(args))

    selected = select_scans(args.data, args.subjects_per_modality, args.seed)
    with (args.output / "sample.jsonl").open("w") as f:
        for row in selected:
            f.write(json.dumps(row) + "\n")
    selected_by_key = {row["key"]: row for row in selected}

    device = torch.device("cuda")
    backbone, _ = load_backbone(CHECKPOINT)
    backbone.to(device).eval().requires_grad_(False)

    embedded_keys = []
    embedding_batches = []
    shard_paths = sorted((args.data / "train").glob("shard.*.tar"))
    for shard_path in tqdm(shard_paths, desc="shards"):
        samples = load_selected_shard(shard_path, selected_by_key)
        for offset in range(0, len(samples), args.batch_size):
            batch_samples = samples[offset : offset + args.batch_size]
            embedding_batches.append(embed_batch(backbone, batch_samples, device))
            embedded_keys.extend(sample["key"] for sample in batch_samples)

    embeddings = np.concatenate(embedding_batches).astype(np.float64)
    assert len(embedded_keys) == len(selected_by_key)
    assert set(embedded_keys) == set(selected_by_key)
    assert embeddings.shape == (len(selected), 1024)

    normalization = np.column_stack([embeddings.mean(axis=0), embeddings.std(axis=0)])
    normalization = normalization.astype(np.float32)
    assert normalization.shape == (1024, 2)
    assert (normalization[:, 1] > 0).all()
    np.save(args.output / "normalization.npy", normalization)
    summary = {
        "checkpoint": CHECKPOINT,
        "pooling": "final_layer_valid_patch_mean",
        "preprocessing": "FOMO300K pretraining-ready sparse tensors",
        "seed": args.seed,
        "subjects_per_modality": args.subjects_per_modality,
        "modalities": list(MODALITIES),
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
