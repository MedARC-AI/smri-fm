---
name: smri-mae-checkpoint
description: sMRI MAE test checkpoint path and load quirks — 3.9GB with optimizer state (use mmap), and args.model_kwargs carries a decoding key the current class rejects.
metadata:
  type: reference
---

`/data/mihir-stuff/smri-pretrained/pretrain_full_90_10_h100/checkpoint-last.pth` — epoch 99,
`mae_vit_large`, patch 8, `img_size` 208x240x208, 26x30x26 patch grid, 1024-d.

- 3.9 GB because the optimizer state rides along, so load with `mmap=True`.
- `args.model_kwargs` carries `decoding: attn`, which today's `MaskedAutoencoderViT` no longer takes
  — filter through `smri_mae.utils.filter_kwargs` or construction raises `TypeError`.
- Weights load strictly: 0 missing, 0 unexpected.

Mihir's in-house 3D ViT MAE over the vendored `src/smri_mae/`. Not a fork — the training code is in
the repo, so there is no upstream inference pipeline to equivalence-test the transform against.

See [[smri-mae-preprocessing-gap]], [[nested-tensor-sdpa-needs-a-device]].
