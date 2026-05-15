# Source this file (do not execute) to set up env vars asparagus needs.
#
#   source scripts/setup_asparagus_env.sh
#
# Values from repo-root .env take precedence over the defaults below.

# Resolve repo root via git (shell-agnostic; works from any cwd inside the repo).
_smri_fm_repo="$(git rev-parse --show-toplevel)"
_smri_fm_env="${_smri_fm_repo}/.env"

# Where asparagus looks for our Hydra config overlays.
# All four flavors point at the same dir; subpaths (model/, task/, ...) disambiguate.
ASPARAGUS_FINETUNE_CONFIGS="${_smri_fm_repo}/src/asparagus_bridge/configs"
ASPARAGUS_TRAIN_CONFIGS="${_smri_fm_repo}/src/asparagus_bridge/configs"
ASPARAGUS_PRETRAIN_CONFIGS="${_smri_fm_repo}/src/asparagus_bridge/configs"
ASPARAGUS_EVAL_BOX_CONFIGS="${_smri_fm_repo}/src/asparagus_bridge/configs"

# ASPARAGUS_CONFIGS is asparagus' *primary* Hydra config path (where its default_*.yaml live).
# This must point at the asparagus submodule's configs/ directory; our overlay is layered on top
# via the *_CONFIGS plural variants above.
ASPARAGUS_CONFIGS="${_smri_fm_repo}/third_party/asparagus/configs"

# Data / model / results / raw-labels paths. Put overrides in repo-root .env.
ASPARAGUS_SOURCE="${_smri_fm_repo}/data/asparagus/source"
ASPARAGUS_DATA="${_smri_fm_repo}/data/asparagus/data"
ASPARAGUS_MODELS="${_smri_fm_repo}/data/asparagus/models"
ASPARAGUS_RESULTS="${_smri_fm_repo}/data/asparagus/results"
ASPARAGUS_RAW_LABELS="${_smri_fm_repo}/data/asparagus/raw_labels"

if [ -f "$_smri_fm_env" ]; then
    # Export every assignment in .env, including project-specific variables that
    # are not part of the ASPARAGUS_* defaults below.
    case "$-" in
        *a*) _smri_fm_restore_allexport=0 ;;
        *) _smri_fm_restore_allexport=1; set -a ;;
    esac
    . "$_smri_fm_env"
    if [ "$_smri_fm_restore_allexport" = "1" ]; then
        set +a
    fi
fi

export ASPARAGUS_FINETUNE_CONFIGS
export ASPARAGUS_TRAIN_CONFIGS
export ASPARAGUS_PRETRAIN_CONFIGS
export ASPARAGUS_EVAL_BOX_CONFIGS
export ASPARAGUS_CONFIGS
export ASPARAGUS_SOURCE
export ASPARAGUS_DATA
export ASPARAGUS_MODELS
export ASPARAGUS_RESULTS
export ASPARAGUS_RAW_LABELS

echo "ASPARAGUS_FINETUNE_CONFIGS=${ASPARAGUS_FINETUNE_CONFIGS}"
echo "ASPARAGUS_TRAIN_CONFIGS=${ASPARAGUS_TRAIN_CONFIGS}"
echo "ASPARAGUS_PRETRAIN_CONFIGS=${ASPARAGUS_PRETRAIN_CONFIGS}"
echo "ASPARAGUS_EVAL_BOX_CONFIGS=${ASPARAGUS_EVAL_BOX_CONFIGS}"
echo "ASPARAGUS_CONFIGS=${ASPARAGUS_CONFIGS}"
echo "ASPARAGUS_SOURCE=${ASPARAGUS_SOURCE}"
echo "ASPARAGUS_DATA=${ASPARAGUS_DATA}"
echo "ASPARAGUS_MODELS=${ASPARAGUS_MODELS}"
echo "ASPARAGUS_RESULTS=${ASPARAGUS_RESULTS}"
echo "ASPARAGUS_RAW_LABELS=${ASPARAGUS_RAW_LABELS}"

unset _smri_fm_repo _smri_fm_env _smri_fm_restore_allexport
