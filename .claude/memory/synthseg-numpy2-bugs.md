---
name: synthseg-numpy2-bugs
description: numpy 2 broke the SynthSeg port in three places, all a shape-(1,) np.where result assigned into a scalar slot; the numpy<2 pin belongs to TensorFlow.
metadata:
  type: project
  observed: 2026-07-29
---

Three failures in the `MedARC-AI/SynthSeg` @ `pytorch-port` fork, all the same shape — a shape-(1,)
`np.where` result assigned into a scalar slot:

- one in `align_volume_to_ref`, firing only for volumes whose axis *order* is not x/y/z (LIA, ASL —
  most raw T1w), so it hides on well-behaved data;
- three in `get_flip_indices`, firing **unconditionally**, so the port could not construct a
  predictor at all.

The `numpy<2` pin was load-bearing but belongs to TensorFlow (<2.16); it moved to the `tensorflow`
extra.

**Why:** running the fork's own suite found the unconditional one; the conditional one was only
found by reading, which is the general lesson.
**How to apply:** on a numpy-2 migration of vendored scientific code, grep for `np.where` results
flowing into scalar assignments, and don't trust a green suite to have exercised the reorientation
paths. See [[synthseg-integration]].
