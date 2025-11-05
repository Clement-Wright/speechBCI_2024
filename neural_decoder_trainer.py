from copy import deepcopy
from typing import Any, Dict, List, Optional

from neural_decoder import load_state_dict_compat
from neural_decoder.model import GRUDecoder
from neural_decoder.dataset import SpeechDataModule
from neural_decoder.callbacks import (
    CurriculumStageLogger,
    LanguageModelFusionCallback,
    TimerCallback,
)
from neural_decoder.augmentations import build_augmentations
from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
import torch
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# local modules

torch.set_float32_matmul_precision("medium")


def _to_container(config: Any) -> Dict:
    if isinstance(config, DictConfig):
        return OmegaConf.to_container(config, resolve=True)
    return deepcopy(config)


def _normalise_stage_definition(stage_def: Any) -> Dict:
    if isinstance(stage_def, DictConfig):
        return OmegaConf.to_container(stage_def, resolve=True)
    if isinstance(stage_def, dict):
        return dict(stage_def)
    raise TypeError(f"Unsupported stage definition type: {type(stage_def)!r}")


def _load_stage_overrides(stage_def: Dict) -> DictConfig:
    overrides = OmegaConf.create({})

    config_path = stage_def.get("config")
    if config_path:
        cfg_path = to_absolute_path(config_path)
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"Curriculum stage config not found: {cfg_path}")
        overrides = OmegaConf.merge(overrides, OmegaConf.load(cfg_path))

    if "overrides" in stage_def and stage_def["overrides"] is not None:
        overrides = OmegaConf.merge(overrides, OmegaConf.create(stage_def["overrides"]))

    return overrides


def _safe_stage_dir_name(index: int, name: str) -> str:
    safe_name = name.lower().replace(" ", "_") if name else f"stage_{index + 1}"
    safe_name = "".join(ch for ch in safe_name if ch.isalnum() or ch in {"_", "-"})
    if not safe_name:
        safe_name = f"stage_{index + 1}"
    return f"stage{index + 1:02d}_{safe_name}"


def _plot_curriculum_metrics(stage_results: List[Dict], output_dir: str) -> Optional[str]:
    metric_candidates = ("accuracy", "wer", "cer", "ser")
    metrics: Dict[str, List[float]] = {}
    stage_labels: List[str] = []

    for idx, result in enumerate(stage_results):
        stage_info = result.get("stage_info") or {}
        label = stage_info.get("name", f"stage_{idx + 1}")
        stage_labels.append(label)
        metric_values = result.get("metrics", {})
        for name, value in metric_values.items():
            if not name.startswith("val/"):
                continue
            if not any(token in name for token in metric_candidates):
                continue
            metrics.setdefault(name, []).append(value)

    if not metrics:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover - optional dependency
        print("matplotlib not available; skipping curriculum metric plot generation.")
        return None

    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = list(range(1, len(stage_labels) + 1))

    for metric_name, values in metrics.items():
        ax.plot(x[: len(values)], values, marker="o", label=metric_name)

    ax.set_xlabel("Curriculum Stage")
    ax.set_ylabel("Metric Value")
    ax.set_xticks(x)
    ax.set_xticklabels(stage_labels, rotation=30, ha="right")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    ax.set_title("Curriculum Validation Metrics")

    plot_path = os.path.join(output_dir, "curriculum_metrics.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def trainModel(args, stage_info: Optional[Dict[str, Any]] = None):
    args = _to_container(args)
    stage_context = dict(stage_info) if stage_info else {}

    # set seed
    pl.seed_everything(args["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    wandb_cfg = args.get("wandb") or {}
    wandb_enabled = bool(wandb_cfg.get("enabled"))

    base_output_dir = args.get("outputDir", os.getcwd())
    stage_directory = stage_context.get("directory")
    if stage_info and not stage_directory:
        stage_directory = _safe_stage_dir_name(
            stage_context.get("index", 0), stage_context.get("name", "stage")
        )
    if stage_directory:
        base_output_dir = os.path.join(base_output_dir, stage_directory)
        stage_context["directory"] = stage_directory

    run_suffix = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    if wandb_enabled and local_rank == 0 and getattr(wandb, "run", None) is not None:
        run_output_dir = os.path.join(base_output_dir, wandb.run.name)
    else:
        run_output_dir = os.path.join(base_output_dir, run_suffix)

    args["outputDir"] = run_output_dir
    stage_context["output_root"] = base_output_dir

    if local_rank == 0:
        os.makedirs(run_output_dir, exist_ok=True)
        with open(os.path.join(run_output_dir, "args"), "wb") as file:
            pickle.dump(args, file)

    dataset_path = to_absolute_path(args["datasetPath"])
    with open(dataset_path, "rb") as handle:
        loadedData = pickle.load(handle)

    train_transform = build_augmentations(
        args.get("trainTransforms"), input_channels=args.get("nInputFeatures")
    )
    eval_transform = build_augmentations(
        args.get("evalTransforms"), input_channels=args.get("nInputFeatures")
    )

    dataset_options = args.get("datasetOptions", {})
    dm = SpeechDataModule(
        loadedData,
        args["batchSize"],
        args["numWorkers"],
        train_transform=train_transform,
        eval_transform=eval_transform,
        dataset_options=dataset_options,
    )

    decoder_cfg = args.get("decoder") or {}
    lm_decoder_config = decoder_cfg.get("lm_decoder") if isinstance(decoder_cfg, dict) else None
    fusion_cfg = decoder_cfg.get("fusion") if isinstance(decoder_cfg, dict) else None

    logger = TensorBoardLogger(run_output_dir, name="torch_dist_v0")

    model = GRUDecoder(
        neural_dim=args["nInputFeatures"],
        n_classes=args["nClasses"],
        hidden_dim=args["nUnits"],
        layer_dim=args["nLayers"],
        nDays=len(loadedData["train"]),
        dropout=args["dropout"],
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"],
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        l2_decay=args["l2_decay"],
        lrStart=args["lrStart"],
        lrEnd=args["lrEnd"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        nBatch=args["nSteps"],
        output_dir=run_output_dir,
        conv_kernel_sizes=list(args["model"]["conv_kernel_sizes"]),
        conv_dilations=list(args["model"]["conv_dilations"]),
        attention_heads=args["model"]["attention_heads"],
        attention_dropout=args["model"].get("attention_dropout", 0.0),
        lm_decoder_config=lm_decoder_config,
    )

    model.curriculum_stage = stage_context

    init_checkpoint = args.get("initCheckpoint")
    if init_checkpoint:
        ckpt_path = to_absolute_path(init_checkpoint)
        if os.path.exists(ckpt_path):
            state = torch.load(ckpt_path, map_location="cpu")
            state_dict = state.get("state_dict", state)
            load_state_dict_compat(model, state_dict)

    checkpointCallback = ModelCheckpoint(
        filename=os.path.join(run_output_dir, "modelWeights"),
        monitor="val/ser",
        mode="min",
        save_top_k=1,
        every_n_train_steps=None,
    )
    checkpointCallback.FILE_EXTENSION = ""

    callbacks: List[Callback] = [checkpointCallback, TimerCallback()]
    if fusion_cfg and fusion_cfg.get("enabled", False):
        beam_width = fusion_cfg.get("beam_search_width", fusion_cfg.get("beam_width", 10))
        fusion_weight = fusion_cfg.get("fusion_weight", fusion_cfg.get("weight", 0.3))
        temperature = fusion_cfg.get("temperature", 1.0)
        length_penalty = fusion_cfg.get("length_penalty", 0.0)
        fusion_callback = LanguageModelFusionCallback(
            fusion_weight=fusion_weight,
            beam_width=int(beam_width),
            temperature=float(temperature),
            length_penalty=float(length_penalty),
            lm_model_name=fusion_cfg.get("lm_model"),
            tokenizer_name=fusion_cfg.get("tokenizer"),
            device=fusion_cfg.get("device", "auto"),
            log_metric=fusion_cfg.get("log_metric", "val/lm_cer"),
        )
        callbacks.append(fusion_callback)

    if stage_info is not None:
        callbacks.append(
            CurriculumStageLogger(
                stage_index=stage_context.get("index", 0),
                stage_name=stage_context.get("name", "stage"),
                total_stages=stage_context.get("total_stages")
                or stage_context.get("total", 1),
                metadata=stage_context.get("metadata"),
            )
        )

    trainer = pl.Trainer(
        strategy=DDPStrategy(find_unused_parameters=True),
        logger=logger,
        min_steps=1,
        max_steps=args["nSteps"],
        accelerator=args["accelerator"],
        devices=args["devices"],
        precision=args["precision"],
        num_nodes=1,
        log_every_n_steps=1,
        val_check_interval=100,
        check_val_every_n_epoch=None,
        callbacks=callbacks,
    )

    trainer.fit(model, dm)

    metrics: Dict[str, float] = {}
    for key, value in trainer.callback_metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
        elif torch.is_tensor(value):
            metrics[key] = float(value.detach().cpu().item())

    stage_context["run_output_dir"] = run_output_dir

    return {
        "best_checkpoint": checkpointCallback.best_model_path,
        "output_dir": run_output_dir,
        "stage_info": stage_context,
        "metrics": metrics,
    }


def _run_curriculum(cfg: DictConfig) -> List[Dict]:
    curriculum_cfg = cfg.get("curriculum")
    if curriculum_cfg is None:
        return [trainModel(cfg)]

    stages = curriculum_cfg.get("stages") if isinstance(curriculum_cfg, dict) else curriculum_cfg.stages
    enabled = curriculum_cfg.get("enabled") if isinstance(curriculum_cfg, dict) else curriculum_cfg.enabled
    if not enabled or not stages:
        return [trainModel(cfg)]

    results: List[Dict] = []
    base_cfg = OmegaConf.create(cfg)
    current_cfg = OmegaConf.create(cfg)
    previous_result: Optional[Dict] = None

    for idx, stage in enumerate(stages):
        stage_def = _normalise_stage_definition(stage)
        overrides = _load_stage_overrides(stage_def)

        inherit = stage_def.get("inherit", True)
        warm_start = stage_def.get("warm_start", True)
        stage_name = stage_def.get("name", f"stage_{idx + 1}")
        stage_metadata = stage_def.get("metadata") or stage_def.get("notes")
        stage_directory = stage_def.get("directory") or stage_def.get("output_suffix")

        stage_base = current_cfg if inherit and idx > 0 else base_cfg
        stage_cfg = OmegaConf.merge(stage_base, overrides)
        if "curriculum" in stage_cfg:
            stage_cfg.curriculum.enabled = False

        stage_args = _to_container(stage_cfg)
        if warm_start and previous_result and previous_result.get("best_checkpoint"):
            stage_args["initCheckpoint"] = previous_result["best_checkpoint"]

        stage_info = {
            "index": idx,
            "name": stage_name,
            "total_stages": len(stages),
            "metadata": stage_metadata,
        }
        if stage_directory:
            stage_info["directory"] = stage_directory

        result = trainModel(stage_args, stage_info=stage_info)
        results.append(result)
        current_cfg = stage_cfg
        previous_result = result

    return results


def loadModel(modelWeightPath, nInputLayers=24, device="cuda"):

    # load pl model
    pl_model = torch.load(modelWeightPath, map_location=device)

    # load hyperparameters
    args = pl_model["hyper_parameters"]
    state_dict = pl_model["state_dict"]

    model = GRUDecoder(
        neural_dim=args["neural_dim"],
        n_classes=args["n_classes"],
        hidden_dim=args["hidden_dim"],
        layer_dim=args["layer_dim"],
        nDays=nInputLayers,
        dropout=args["dropout"],
        strideLen=args["strideLen"],
        kernelLen=args["kernelLen"],
        gaussianSmoothWidth=args["gaussianSmoothWidth"],
        whiteNoiseSD=args["whiteNoiseSD"],
        constantOffsetSD=args["constantOffsetSD"],
        bidirectional=args["bidirectional"],
        l2_decay=args["l2_decay"],
        lrStart=args["lrStart"],
        lrEnd=args["lrEnd"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        gamma=args["gamma"],
        stepSize=args["stepSize"],
        nBatch=args["nSteps"],
        output_dir=args.get("output_dir", args.get("outputDir", "")),
        conv_kernel_sizes=args.get("conv_kernel_sizes"),
        conv_dilations=args.get("conv_dilations"),
        attention_heads=args.get("attention_heads", 4),
        attention_dropout=args.get("attention_dropout", 0.0),
        lm_decoder_config=args.get("lm_decoder_config"),
    ).to(device)

    load_state_dict_compat(model, state_dict)
    return model


@hydra.main(version_base="1.1", config_path="conf", config_name="config_1")
def main(cfg):

    # local rank for distributed training
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    conf_name = HydraConfig.get().job.config_name
    wandb_run = None
    wandb_enabled = bool(getattr(cfg, "wandb", None) and cfg.wandb.enabled)

    if wandb_enabled:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        if "_" in conf_name:
            wandb_config["hyperparam_setting"] = conf_name[conf_name.index("_") + 1 :]
        run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S-')}" + \
            wandb_config.get("hyperparam_setting", conf_name)

        print(f"config: {conf_name[conf_name.index('_')+1:]}" if "_" in conf_name else f"config: {conf_name}")
        if local_rank == 0:
            wandb_run = wandb.init(
                **cfg.wandb.setup,
                config=wandb_config,
                name=run_name,
                sync_tensorboard=True,
            )

    stage_results = _run_curriculum(cfg)

    curriculum_node = getattr(cfg, "curriculum", None)
    if isinstance(curriculum_node, DictConfig):
        curriculum_enabled = bool(curriculum_node.get("enabled"))
    elif isinstance(curriculum_node, dict):
        curriculum_enabled = bool(curriculum_node.get("enabled"))
    else:
        curriculum_enabled = False
    if (
        local_rank == 0
        and curriculum_enabled
        and stage_results
    ):
        last_stage_info = stage_results[-1].get("stage_info", {})
        plot_dir = last_stage_info.get("output_root") or stage_results[-1].get("output_dir")
        plot_path = _plot_curriculum_metrics(stage_results, plot_dir)
        if plot_path:
            print(f"Curriculum metrics plot saved to {plot_path}")

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
