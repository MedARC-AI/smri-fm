# Source this file (do not execute) to set up env vars asparagus needs.
#
#   source scripts/setup_asparagus_env.sh
#
# Override any of these by exporting them before sourcing.

# Resolve repo root via git (shell-agnostic; works from any cwd inside the repo).
_smri_fm_repo="$(git rev-parse --show-toplevel)"

# Where asparagus looks for our Hydra config overlays.
# All four flavors point at the same dir; subpaths (model/, task/, ...) disambiguate.
: "${ASPARAGUS_FINETUNE_CONFIGS:=${_smri_fm_repo}/src/asparagus_bridge/configs}"
: "${ASPARAGUS_TRAIN_CONFIGS:=${_smri_fm_repo}/src/asparagus_bridge/configs}"
: "${ASPARAGUS_PRETRAIN_CONFIGS:=${_smri_fm_repo}/src/asparagus_bridge/configs}"
: "${ASPARAGUS_EVAL_BOX_CONFIGS:=${_smri_fm_repo}/src/asparagus_bridge/configs}"
export ASPARAGUS_FINETUNE_CONFIGS ASPARAGUS_TRAIN_CONFIGS ASPARAGUS_PRETRAIN_CONFIGS ASPARAGUS_EVAL_BOX_CONFIGS

# ASPARAGUS_CONFIGS is asparagus' *primary* Hydra config path (where its default_*.yaml live).
# This must point at the asparagus submodule's configs/ directory; our overlay is layered on top
# via the *_CONFIGS plural variants above.
: "${ASPARAGUS_CONFIGS:=${_smri_fm_repo}/third_party/asparagus/configs}"

# Data / model / results / raw-labels paths. Override these to point at real shared storage.
: "${ASPARAGUS_DATA:=${_smri_fm_repo}/data/asparagus/data}"
: "${ASPARAGUS_MODELS:=${_smri_fm_repo}/data/asparagus/models}"
: "${ASPARAGUS_RESULTS:=${_smri_fm_repo}/data/asparagus/results}"
: "${ASPARAGUS_RAW_LABELS:=${_smri_fm_repo}/data/asparagus/raw_labels}"
export ASPARAGUS_DATA ASPARAGUS_MODELS ASPARAGUS_RESULTS ASPARAGUS_RAW_LABELS ASPARAGUS_CONFIGS

unset _smri_fm_repo
