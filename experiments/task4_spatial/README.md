# `task4_spatial`

Test effect of train/test time spatial augmentation.

## Results

| run | train views | tta views | alpha | Dice | 95% CI | nerve | vessel | oracle | nerve cut | vessel cut | paired vs base | min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| train-0_test-0_alpha-1e1 | 1 | 1 | 10 | **0.354** | [0.301, 0.400] | 0.433 | 0.275 | 0.384 | 1.6e-01 | 1.6e-01 | +0.000 [+0.000, +0.000] | 8 |
| train-0_test-0_alpha-3.3 | 1 | 1 | 3.3 | **0.351** | [0.300, 0.395] | 0.431 | 0.270 | 0.376 | 1.7e-01 | 1.7e-01 | -0.003 [-0.006, +0.000] | 8 |
| train-0_test-4_alpha-1e1 | 1 | 5 | 10 | **0.419** | [0.355, 0.472] | 0.494 | 0.344 | 0.460 | 1.6e-01 | 1.3e-01 | +0.065 [+0.050, +0.079] | 10 |
| train-2_test-0_alpha-1e1 | 3 | 1 | 10 | **0.379** | [0.329, 0.425] | 0.462 | 0.297 | 0.410 | 1.6e-01 | 1.7e-01 | +0.026 [+0.018, +0.032] | 27 |
| train-2_test-4_alpha-1e1 | 3 | 5 | 10 | **0.440** | [0.378, 0.491] | 0.518 | 0.362 | 0.481 | 1.7e-01 | 1.4e-01 | +0.086 [+0.071, +0.100] | 28 |
| train-2_test-4_alpha-3e1 | 3 | 5 | 30 | **0.433** | [0.369, 0.486] | 0.508 | 0.358 | 0.475 | 1.7e-01 | 1.4e-01 | +0.079 [+0.064, +0.094] | 21 |

- test time augmentation helps a lot
- train time helps modestly
