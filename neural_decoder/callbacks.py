import logging
from typing import Any, Dict, List, Optional

import torch
from edit_distance import SequenceMatcher
from pytorch_lightning.callbacks import Callback

import time


LOGGER = logging.getLogger(__name__)

class TimerCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.train_batch_time = time.time()
    
    def on_train_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        train_time = time.time() - self.train_batch_time
        pl_module.log(
            "train/batchComputationTime",
            train_time,
            sync_dist=True,
            rank_zero_only=True,
            prog_bar=False,
        )
        self.train_batch_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_epoch_time = time.time()
    
    def on_train_epoch_end(self, trainer, pl_module):
        train_time = time.time() - self.train_epoch_time
        pl_module.log(
            "train/epochComputationTime",
            train_time,
            sync_dist=True,
            rank_zero_only=True,
            prog_bar=False,
        )
        self.train_epoch_time = 0
    
    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self.val_batch_time = time.time()
    
    def on_validation_batch_end(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        val_time = time.time() - self.val_batch_time
        pl_module.log(
            "val/batchComputationTime",
            val_time,
            sync_dist=True,
            rank_zero_only=True,
            prog_bar=False,
        )
        self.val_batch_time = 0


class CurriculumStageLogger(Callback):
    """Log transitions between curriculum stages."""

    def __init__(
        self,
        stage_index: int,
        stage_name: str,
        total_stages: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.stage_index = stage_index
        self.stage_name = stage_name
        self.total_stages = total_stages
        self.metadata = metadata or {}
        self._logged = False

    def setup(self, trainer, pl_module, stage: Optional[str] = None):
        pl_module.curriculum_stage = {
            "index": self.stage_index,
            "name": self.stage_name,
            "total": self.total_stages,
            "metadata": self.metadata,
        }

    def on_train_start(self, trainer, pl_module):
        self._log_stage(trainer, pl_module)

    def on_validation_start(self, trainer, pl_module):
        self._log_stage(trainer, pl_module)

    def _log_stage(self, trainer, pl_module):
        if self._logged:
            return

        LOGGER.info(
            "Starting curriculum stage %s/%s: %s",
            self.stage_index + 1,
            self.total_stages,
            self.stage_name,
        )

        metrics = {
            "curriculum/stage_index": float(self.stage_index),
            "curriculum/stage_number": float(self.stage_index + 1),
            "curriculum/total_stages": float(max(self.total_stages, 1)),
        }

        if trainer.logger is not None:
            try:
                trainer.logger.log_metrics(metrics, step=trainer.global_step)
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("Logger does not support stage metric logging.")

            try:
                trainer.logger.log_hyperparams(
                    {
                        "curriculum_stage_name": self.stage_name,
                        "curriculum_stage_index": self.stage_index,
                        "curriculum_total_stages": self.total_stages,
                    }
                )
            except Exception:  # pragma: no cover - defensive
                LOGGER.debug("Logger does not support hyperparameter logging for curriculum stage.")

        try:
            pl_module.log(
                "curriculum/stage_index",
                float(self.stage_index),
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )
            pl_module.log(
                "curriculum/total_stages",
                float(max(self.total_stages, 1)),
                prog_bar=False,
                sync_dist=True,
                rank_zero_only=True,
            )
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Module logging not available for curriculum stage metrics.")

        self._logged = True


class LanguageModelFusionCallback(Callback):
    """Perform shallow-fusion rescoring with an external language model."""

    def __init__(
        self,
        *,
        fusion_weight: float = 0.3,
        beam_width: int = 10,
        temperature: float = 1.0,
        length_penalty: float = 0.0,
        lm_model_name: Optional[str] = None,
        tokenizer_name: Optional[str] = None,
        device: Optional[str] = "auto",
        log_metric: str = "val/lm_cer",
    ) -> None:
        super().__init__()

        self.fusion_weight = fusion_weight
        self.beam_width = beam_width
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.lm_model_name = lm_model_name
        self.tokenizer_name = tokenizer_name or lm_model_name
        self.device = device
        self.log_metric = log_metric

        self._lm_model = None
        self._lm_tokenizer = None
        self._adapter = None
        self._buffer: List[Dict] = []
        self._char_cache: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------
    def setup(self, trainer, pl_module, stage: Optional[str] = None) -> None:
        self._adapter = getattr(pl_module, "decoder_adapter", None)
        if self._adapter is None:
            LOGGER.warning(
                "LanguageModelFusionCallback attached to module without decoder adapter."
            )
            return

        if self.lm_model_name and self._lm_model is None:
            self._load_lm()

    def on_validation_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch,
        batch_idx,
        dataloader_idx=0,
    ) -> None:
        if self._adapter is None or not self._adapter.backend_available:
            return

        if outputs is None or "pred" not in outputs:
            return

        logits = outputs["pred"].detach()
        _, _, X_len, y_len, _ = batch
        logit_lengths = ((X_len - pl_module.kernelLen) / pl_module.strideLen).to(
            torch.int32
        )

        hypotheses = self._adapter.decode_batch(
            logits,
            logit_lengths,
            beam_width=self.beam_width,
            temperature=self.temperature,
        )

        fused_predictions: List[Dict] = []
        for cand_list in hypotheses:
            fused_predictions.append(self._select_with_fusion(cand_list))

        self._buffer.append(
            {
                "predictions": fused_predictions,
                "targets": batch[1].detach().cpu(),
                "target_lengths": y_len.detach().cpu(),
            }
        )

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not self._buffer:
            return

        total_distance = 0.0
        total_length = 0

        for entry in self._buffer:
            predictions = entry["predictions"]
            targets = entry["targets"]
            target_lengths = entry["target_lengths"]
            for pred, target, tgt_len in zip(predictions, targets, target_lengths):
                target_tokens = target[:tgt_len].tolist()
                pred_tokens = pred.get("tokens") if pred else []
                if pred_tokens is None:
                    pred_tokens = []
                if isinstance(pred_tokens, str):
                    pred_tokens = [ord(c) for c in pred_tokens]
                matcher = SequenceMatcher(a=target_tokens, b=pred_tokens)
                total_distance += matcher.distance()
                total_length += len(target_tokens)

        if total_length > 0:
            cer = total_distance / total_length
            pl_module.log(
                self.log_metric,
                cer,
                prog_bar=True,
                sync_dist=True,
                rank_zero_only=True,
            )

        self._buffer.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_lm(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            LOGGER.warning(
                "transformers not installed; external LM fusion disabled."
            )
            return

        model_name = self.lm_model_name
        tokenizer_name = self.tokenizer_name or model_name
        if model_name is None:
            return

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model.to(self.device)
        model.eval()

        self._lm_model = model
        self._lm_tokenizer = tokenizer

    def _select_with_fusion(self, candidates: List[Dict]) -> Dict:
        if not candidates:
            return {}

        best_candidate: Optional[Dict] = None
        best_score = float("-inf")
        for candidate in candidates[: self.beam_width]:
            sentence = candidate.get("sentence", "")
            acoustic_score = candidate.get("score", 0.0)
            lm_score = self._score_with_lm(sentence) if sentence else 0.0
            fused_score = (
                acoustic_score
                + self.fusion_weight * lm_score
                - self.length_penalty * len(sentence.split())
            )
            if fused_score > best_score:
                best_score = fused_score
                best_candidate = candidate

        return best_candidate or candidates[0]

    def _score_with_lm(self, sentence: str) -> float:
        if self._lm_model is None or self._lm_tokenizer is None:
            return 0.0

        if sentence in self._char_cache:
            return self._char_cache[sentence]

        inputs = self._lm_tokenizer(
            sentence,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        with torch.no_grad():
            outputs = self._lm_model(**inputs)
            logits = torch.log_softmax(outputs.logits[:, :-1, :], dim=-1)
            target = inputs["input_ids"][..., 1:]
            token_log_probs = logits.gather(-1, target.unsqueeze(-1)).squeeze(-1)
            score = token_log_probs.sum().item()

        self._char_cache[sentence] = score
        return score
