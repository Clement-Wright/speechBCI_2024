#!/usr/bin/env bash
set -euo pipefail

CONFIG_NAME=${1:-config_kaggle_2025}
if (($# > 0)); then
  shift
fi

python neural_decoder_trainer.py \
  --config-name "${CONFIG_NAME}" \
  decoder.lm_decoder.enabled=true \
  decoder.fusion.enabled=true \
  decoder.fusion.beam_width=20 \
  decoder.fusion.fusion_weight=0.45 \
  decoder.fusion.temperature=0.8 \
  "$@"
