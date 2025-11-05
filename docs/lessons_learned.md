# Lessons learned for the 2025 Brain-to-Text challenge

This note distills how we extended the CIBR 2024 second-place stack in light of the first-place report and the organiser retrospectives. Use it as a quick orientation for new contributors.

## Data and alignment

- Prioritise consistent feature statistics across datasets. Both 2024 papers stress alignment sensitivity; we therefore normalise every loader through the shared `SpeechDataset` filters so Dryad and Kaggle shards can be mixed with minimal drift. Adjust `datasetOptions` for vocabulary and time limits per stage to catch issues early.
- Stage-wise configs are invaluable when onboarding a new dataset. Begin with the low-vocabulary warm-up stage in `conf/config_kaggle_2025.yaml` and promote length/vocabulary in later stages via `scripts/run_curriculum.sh`. The curriculum plot emitted by `neural_decoder_trainer.py` lets you validate convergence transitions without combing through TensorBoard logs.

## Feature engineering

- The first-place team highlighted the importance of rich spectral features (multi-band Hilbert envelopes and high-gamma power). We mirrored those ideas through reusable transforms in `neural_decoder/augmentations.py`, which makes it easy to compare spectral stacks or add ablations.
- Keep augmentation probabilities explicit in Hydra configs so you can sweep time-warp, channel dropout, and noise without editing code. This also matches the reproducibility emphasis in the organiser write-up.

## Model architecture

- We replaced the single-branch convolution with parallel temporal branches that cover short, medium, and long contexts before attention. This design mirrors the champion’s focus on modelling multiple articulatory time scales and feeds a shared `torch.nn.MultiheadAttention` block for sequence reasoning.
- A lightweight squeeze-excite style gating module before concatenation proved critical for stabilising fusion of the parallel streams. It keeps the training dynamics smooth when augmentations change the per-channel variance profile.

## Language model integration

- Shallow fusion with an external LM is mandatory for competitive leaderboard results. The adapter in `neural_decoder/model.py` now bridges our encoder to the vendored Stanford `LanguageModelDecoder`, while `LanguageModelFusionCallback` handles interpolation weights and logging.
- `scripts/run_lm_fusion.sh` demonstrates a good starting point: enable the Kaldi WFST decoder, tune beam width/temperature, and specify the HF model you want to fuse (we found GPT-style checkpoints stable when paired with curriculum fine-tuning).

## Evaluation workflow

- `eval_competition.py` can now emit Kaggle-ready CSV/JSON submissions and optional ZIP bundles. Use `--decoder-mode language_model` for LM-assisted decoding and collect the reported WER/CER in `metrics/` for the comparison notebook.
- The `notebooks/WER_CER_comparison.ipynb` template keeps a running scoreboard between baseline and enhanced runs. Save JSON metric dumps from the trainer or evaluation script and refresh the notebook before each milestone.

## Strategic takeaways

- Curriculum scheduling and LM fusion should be treated as first-class citizens in experiment tracking. When results regress, inspect the curriculum stage logs first—they often reveal misconfigured vocabularies or missing augmentation seeds.
- The 2024 retrospectives emphasise ensembling, but our experience shows that a single curriculum-trained, LM-fused model can reach competitive numbers if the spectral front-end and gating stack are tuned carefully. Focus ensemble efforts only after the single-model pipeline saturates.
