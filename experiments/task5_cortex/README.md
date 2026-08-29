# `task5_cortex`

```bash
sbatch experiments/task5_cortex/launch.sh                # 8 runs, ~140s each
uv run python experiments/task5_cortex/collect.py        # the table below
uv run python experiments/task5_cortex/plot_scores.py    # figures/scores.png
```

## Results

| ckpt | pooling | cortex_frac | AUROC | 95% CI | time |
|---|---|---|---|---|---|
| pt-full | global | -- | **0.898** | 0.790 – 0.979 | 150s |
| pt-full | cortex | 0 | **0.917** | 0.824 – 0.984 | 150s |
| pt-full | cortex | 0.1 | **0.936** | 0.859 – 0.989 | 152s |
| pt-full | cortex | 0.25 | **0.938** | 0.861 – 0.990 | 134s |
| walnut-vitl | global | -- | **0.882** | 0.774 – 0.972 | 131s |
| walnut-vitl | cortex | 0 | **0.948** | 0.883 – 0.992 | 138s |
| walnut-vitl | cortex | 0.1 | **0.951** | 0.886 – 0.993 | 127s |
| walnut-vitl | cortex | 0.25 | **0.941** | 0.869 – 0.990 | 128s |
