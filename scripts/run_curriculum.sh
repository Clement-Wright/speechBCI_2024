#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME=${1:-config_kaggle_2025}
if (($# > 0)); then
  shift
fi

python neural_decoder_trainer.py \
  --config-name "${CONFIG_NAME}" \
  curriculum.enabled=true \
  "$@"
