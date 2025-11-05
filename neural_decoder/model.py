import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from edit_distance import SequenceMatcher
from torch import nn, optim

from third_party import speechBCI as speechbci_runtime

from .augmentations import GaussianSmoothing


LOGGER = logging.getLogger(__name__)


class ChannelGate(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, channels)

    def forward(self, x):
        # x: (B, C, T)
        squeeze = x.mean(dim=-1)
        excitation = torch.sigmoid(self.fc2(F.relu(self.fc1(squeeze))))
        return x * excitation.unsqueeze(-1)


class LanguageModelDecoderAdapter(nn.Module):
    """Project encoder states and optionally decode with speechBCI's LM."""

    DEFAULT_DECODE_OPTIONS = {
        "max_active": 7000,
        "min_active": 200,
        "beam": 17.0,
        "lattice_beam": 8.0,
        "acoustic_scale": 1.0,
        "ctc_blank_skip_threshold": 0.98,
        "nbest": 10,
    }

    def __init__(
        self,
        encoder_dim: int,
        vocab_size: int,
        blank_id: int = 0,
        lm_decoder_config: Optional[Dict] = None,
    ) -> None:
        super().__init__()

        self.projection = nn.Linear(encoder_dim, vocab_size)
        self.blank_id = blank_id

        self._config = lm_decoder_config.copy() if lm_decoder_config else {}
        self._backend = None
        self._decode_resource = None
        self._runtime_root: Optional[Path] = None
        self._resource_paths: Dict[str, str] = {}
        self._decode_options_dict: Dict[str, float] = {}
        self._logit_mapping: Optional[Iterable[int]] = None

        if self._config.get("enabled", False):
            self._initialise_backend()

    @property
    def backend_available(self) -> bool:
        return self._backend is not None and self._decode_resource is not None

    @property
    def decode_options(self) -> Dict[str, float]:
        if not self._decode_options_dict:
            return self.DEFAULT_DECODE_OPTIONS.copy()
        return self._decode_options_dict.copy()

    @property
    def logit_mapping(self) -> Optional[Iterable[int]]:
        return self._logit_mapping

    def forward(self, encoder_states: torch.Tensor) -> torch.Tensor:
        return self.projection(encoder_states)

    # ------------------------------------------------------------------
    # Backend initialisation
    # ------------------------------------------------------------------
    def _initialise_backend(self) -> None:
        runtime_root = self._config.get("runtime_root")
        auto_download = bool(self._config.get("auto_download", False))

        if runtime_root is None:
            runtime_root_path = speechbci_runtime.ensure_runtime(
                download=auto_download
            )
        else:
            runtime_root_path = Path(runtime_root)

        self._runtime_root = runtime_root_path

        if not runtime_root_path.exists():
            LOGGER.warning(
                "LanguageModelDecoder runtime directory %s does not exist.",
                runtime_root_path,
            )
            return

        search_paths: List[Path] = []
        python_dir = runtime_root_path / "python"
        if python_dir.exists():
            search_paths.append(python_dir)

        build_dir = runtime_root_path / "build"
        if build_dir.exists():
            search_paths.extend(
                p
                for p in build_dir.rglob("*")
                if p.is_dir() and ("lib" in p.name or "Release" in p.name)
            )

        for path in search_paths:
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)

        try:
            self._backend = importlib.import_module("lm_decoder")
        except ImportError:
            LOGGER.warning(
                "Unable to import lm_decoder extension. Proceeding without the "
                "external WFST decoder."
            )
            self._backend = None
            return

        resource_cfg = self._config.get("resource", {})
        self._resource_paths = {
            "fst_path": resource_cfg.get("fst_path", ""),
            "const_arpa_path": resource_cfg.get("const_arpa_path", ""),
            "g_path": resource_cfg.get("g_path", ""),
            "words_path": resource_cfg.get("words_path", ""),
            "symbol_table": resource_cfg.get("symbol_table", ""),
        }

        missing = [k for k, v in self._resource_paths.items() if not v]
        if missing:
            LOGGER.warning(
                "LanguageModelDecoder resources missing (%s); backend disabled.",
                ", ".join(missing),
            )
            self._backend = None
            return

        try:
            self._decode_resource = self._backend.DecodeResource(
                self._resource_paths["fst_path"],
                self._resource_paths["const_arpa_path"],
                self._resource_paths["g_path"],
                self._resource_paths["words_path"],
                self._resource_paths["symbol_table"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.warning("Failed to load LM decode resources: %s", exc)
            self._backend = None
            self._decode_resource = None
            return

        options_cfg = self._config.get("decode_options", {})
        self._decode_options_dict = self.DEFAULT_DECODE_OPTIONS.copy()
        self._decode_options_dict.update(options_cfg)
        self._logit_mapping = self._config.get("logit_mapping")

    def _create_decode_options(self, overrides: Optional[Dict] = None):
        if self._backend is None:
            return None

        options = self.decode_options
        if overrides:
            options.update(overrides)

        return self._backend.DecodeOptions(
            int(options["max_active"]),
            int(options["min_active"]),
            float(options["beam"]),
            float(options["lattice_beam"]),
            float(options["acoustic_scale"]),
            float(options["ctc_blank_skip_threshold"]),
            int(options["nbest"]),
        )

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------
    def decode_batch(
        self,
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None,
        *,
        beam_width: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[List[Dict]]:
        if not self.backend_available:
            return [[] for _ in range(logits.shape[0])]

        decode_options = self._create_decode_options(
            {"nbest": beam_width} if beam_width is not None else None
        )
        decoder = self._backend.BrainSpeechDecoder(
            self._decode_resource, decode_options
        )

        logits_np = logits.detach().cpu().numpy()
        if logit_lengths is not None:
            logit_lengths = logit_lengths.detach().cpu().numpy()

        hypotheses: List[List[Dict]] = []
        for batch_idx in range(logits_np.shape[0]):
            cur_logits = logits_np[batch_idx]
            if logit_lengths is not None:
                cur_logits = cur_logits[: int(logit_lengths[batch_idx])]

            if temperature != 1.0:
                cur_logits = cur_logits / max(temperature, 1e-5)

            if self._logit_mapping:
                cur_logits = cur_logits[:, list(self._logit_mapping)]

            self._backend.DecodeNumpy(decoder, cur_logits)
            decoder.FinishDecoding()

            batch_hyps: List[Dict] = []
            for result in decoder.result():
                hyp = {
                    "sentence": getattr(result, "sentence", ""),
                    "words": getattr(result, "words", []),
                    "tokens": getattr(result, "tokens", None),
                    "score": float(getattr(result, "score", 0.0)),
                }
                if hyp["tokens"] is None and isinstance(hyp["words"], Iterable):
                    try:
                        hyp["tokens"] = [int(w) for w in hyp["words"]]
                    except Exception:  # pragma: no cover - robustness guard
                        hyp["tokens"] = hyp["words"]
                batch_hyps.append(hyp)

            hypotheses.append(batch_hyps)
            decoder.Reset()

        return hypotheses

    def configure_backend(
        self,
        *,
        resource: Optional[Dict[str, str]] = None,
        decode_options: Optional[Dict[str, float]] = None,
        logit_mapping: Optional[Iterable[int]] = None,
    ) -> None:
        """Update backend configuration and reinitialise the decoder."""

        if resource:
            config_resource = self._config.setdefault("resource", {})
            config_resource.update(resource)
        if decode_options:
            config_opts = self._config.setdefault("decode_options", {})
            config_opts.update(decode_options)
        if logit_mapping is not None:
            self._config["logit_mapping"] = list(logit_mapping)

        if resource or decode_options or logit_mapping is not None:
            self._config["enabled"] = True

        if self._config.get("enabled", False):
            self._initialise_backend()


class GRUDecoder(pl.LightningModule):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays,
        dropout,
        strideLen,
        kernelLen,
        gaussianSmoothWidth,
        whiteNoiseSD,
        constantOffsetSD,
        bidirectional,
        l2_decay,
        lrStart,
        lrEnd,
        momentum,
        nesterov,
        gamma,
        stepSize,
        nBatch,
        output_dir,
        conv_kernel_sizes=None,
        conv_dilations=None,
        attention_heads=4,
        attention_dropout=0.0,
        lm_decoder_config: Optional[Dict] = None,
    ):
        super().__init__()

        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3, 7, 15]
        if conv_dilations is None:
            conv_dilations = [1] * len(conv_kernel_sizes)
        if len(conv_kernel_sizes) != len(conv_dilations):
            raise ValueError(
                "conv_kernel_sizes and conv_dilations must have the same length"
            )
        if attention_heads <= 0:
            raise ValueError("attention_heads must be > 0")

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1),
            dilation=1,
            padding=0,
            stride=(self.strideLen, 1),
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(
            torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))
        self.constantOffsetSD = constantOffsetSD
        self.whiteNoiseSD = whiteNoiseSD
        self.loss_ctc = torch.nn.CTCLoss(
            blank=0, reduction="mean", zero_infinity=True)
        self.l2_decay = l2_decay
        self.lrStart = lrStart
        self.lrEnd = lrEnd
        self.momentum = momentum
        self.nesterov = nesterov
        self.gamma = gamma
        self.stepSize = stepSize
        self.nBatch = nBatch
        self.output_dir = output_dir
        self.conv_kernel_sizes = list(conv_kernel_sizes)
        self.conv_dilations = list(conv_dilations)
        self.attention_heads = attention_heads
        self.attention_dropout = attention_dropout
        self.testLoss = []
        self.testCER = []

        self.save_hyperparameters(
            {
                "neural_dim": neural_dim,
                "n_classes": n_classes,
                "hidden_dim": hidden_dim,
                "layer_dim": layer_dim,
                "nDays": nDays,
                "dropout": dropout,
                "strideLen": strideLen,
                "kernelLen": kernelLen,
                "gaussianSmoothWidth": gaussianSmoothWidth,
                "whiteNoiseSD": whiteNoiseSD,
                "constantOffsetSD": constantOffsetSD,
                "bidirectional": bidirectional,
                "l2_decay": l2_decay,
                "lrStart": lrStart,
                "lrEnd": lrEnd,
                "momentum": momentum,
                "nesterov": nesterov,
                "gamma": gamma,
                "stepSize": stepSize,
                "nSteps": nBatch,
                "output_dir": output_dir,
                "conv_kernel_sizes": conv_kernel_sizes,
                "conv_dilations": conv_dilations,
                "attention_heads": attention_heads,
                "attention_dropout": attention_dropout,
                "lm_decoder_config": lm_decoder_config,
            }
        )
        self.branch_out_channels = neural_dim
        self.conv_branches = nn.ModuleList()
        self.branch_gates = nn.ModuleList()
        for kernel_size, dilation in zip(
            self.conv_kernel_sizes, self.conv_dilations
        ):
            padding = ((kernel_size - 1) // 2) * dilation
            branch = nn.Conv1d(
                neural_dim,
                self.branch_out_channels,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=False,
            )
            self.conv_branches.append(branch)
            self.branch_gates.append(ChannelGate(self.branch_out_channels))

        self.total_branch_channels = (
            self.branch_out_channels * len(self.conv_branches)
        )
        if self.total_branch_channels % self.attention_heads != 0:
            raise ValueError(
                "total_branch_channels must be divisible by attention_heads"
            )
        self.attention = nn.MultiheadAttention(
            embed_dim=self.total_branch_channels,
            num_heads=self.attention_heads,
            dropout=self.attention_dropout,
            batch_first=False,
        )
        self.attention_norm = nn.LayerNorm(self.total_branch_channels)

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # GRU layers
        self.gru_decoder = nn.GRU(
            self.total_branch_channels * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.gru_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(
                self, "inpLayer" + str(x),
                nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        decoder_input_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.decoder_adapter = LanguageModelDecoderAdapter(
            encoder_dim=decoder_input_dim,
            vocab_size=n_classes + 1,
            blank_id=0,
            lm_decoder_config=lm_decoder_config,
        )

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        conv_input = torch.permute(transformedNeural, (0, 2, 1))
        branch_features = []
        for branch, gate in zip(self.conv_branches, self.branch_gates):
            branch_out = branch(conv_input)
            branch_out = F.relu(branch_out)
            branch_out = gate(branch_out)
            branch_features.append(branch_out)

        multi_scale_features = torch.cat(branch_features, dim=1)
        attn_in = torch.permute(multi_scale_features, (2, 0, 1))
        attn_out, _ = self.attention(attn_in, attn_in, attn_in)
        attn_out = torch.permute(attn_out, (1, 0, 2))
        attn_out = self.attention_norm(attn_out)
        attn_out = torch.permute(attn_out, (0, 2, 1))

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(attn_out.unsqueeze(-1)),
            (0, 2, 1),
        )

        # apply RNN layer
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * 2,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim,
                transformedNeural.size(0),
                self.hidden_dim,
                device=self.device,
            ).requires_grad_()

        hid, _ = self.gru_decoder(stridedInputs, h0.detach())

        # get seq
        seq_out = self.decoder_adapter(hid)
        return seq_out

    def decode_with_language_model(
        self,
        logits: torch.Tensor,
        logit_lengths: Optional[torch.Tensor] = None,
        *,
        beam_width: Optional[int] = None,
        temperature: float = 1.0,
    ) -> List[List[Dict]]:
        """Run the optional WFST decoder on a batch of logits."""

        if not hasattr(self, "decoder_adapter") or self.decoder_adapter is None:
            raise RuntimeError("Model does not expose a language model adapter.")
        return self.decoder_adapter.decode_batch(
            logits,
            logit_lengths,
            beam_width=beam_width,
            temperature=temperature,
        )

    def training_step(self, batch, batch_idx):
        X, y, X_len, y_len, dayIdx = batch

        # Noise augmentation is faster on GPU
        if self.whiteNoiseSD > 0:
            X += torch.randn(X.shape, device=self.device) * self.whiteNoiseSD

        if self.constantOffsetSD > 0:
            X += (
                torch.randn([X.shape[0], 1, X.shape[2]], device=self.device)
                * self.constantOffsetSD
            )

        # Compute prediction error
        pred = self.forward(X, dayIdx)
        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.kernelLen) / self.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)
        schedulers = self.lr_schedulers()
        cur_lr = schedulers.optimizer.param_groups[-1]["lr"]
        self.log_dict(
            {"train/predictionLoss": loss, "train/learning_rate": cur_lr},
            on_step=True, on_epoch=False, prog_bar=True, sync_dist=True,
            rank_zero_only=True)
        return {"loss": loss, "pred": pred, "y": y}

    def validation_step(self, batch, batch_idx):
        X, y, X_len, y_len, testDayIdx = batch

        total_edit_distance = 0
        total_seq_length = 0

        pred = self.forward(X, testDayIdx)
        loss = self.loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - self.kernelLen) / self.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)

        adjustedLens = ((X_len - self.kernelLen) / self.strideLen).to(
            torch.int32
        )
        for iterIdx in range(pred.shape[0]):
            decodedSeq = torch.argmax(
                pred[iterIdx, 0: adjustedLens[iterIdx], :].clone().detach(),
                dim=-1,
            )  # [num_seq,]
            decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(
                y[iterIdx][0: y_len[iterIdx]].cpu().detach()
            )

            matcher = SequenceMatcher(
                a=trueSeq.tolist(), b=decodedSeq.tolist()
            )
            total_edit_distance += matcher.distance()
            total_seq_length += len(trueSeq)

        avgDayLoss = loss
        cer = total_edit_distance / total_seq_length

        self.log_dict({"val/predictionLoss": avgDayLoss, "val/ser": cer},
                      sync_dist=True, prog_bar=True, rank_zero_only=True)

        return {"loss": loss, "pred": pred, "y": y}

    def configure_optimizers(self):

        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lrStart,
            momentum=self.momentum,
            nesterov=self.nesterov,
            weight_decay=self.l2_decay,
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.stepSize,
            gamma=self.gamma,
        )

        freq = self.trainer.accumulate_grad_batches or 1

        return {
            "optimizer": optimizer,
            "lr_scheduler":
            {
                "scheduler": scheduler,
                "interval": "step",
                "freqeuncy": freq
            }
        }
