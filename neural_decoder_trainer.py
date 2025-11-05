from neural_decoder import load_state_dict_compat
from neural_decoder.model import GRUDecoder
from neural_decoder.dataset import SpeechDataModule
from neural_decoder.callbacks import LanguageModelFusionCallback, TimerCallback
from datetime import datetime
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
import os
import pickle

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

import sys
import torch
import wandb

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
# local modules

torch.set_float32_matmul_precision("medium")


def trainModel(args):

    # set seed
    pl.seed_everything(args["seed"], workers=True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args["seed"])
    torch.backends.cudnn.deterministic = True

    # output directory
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if 'wandb' in args and args.wandb.enabled and local_rank == 0:
        args["outputDir"] = os.path.join(args["outputDir"], wandb.run.name)
    else:
        args["outputDir"] = os.path.join(
            args["outputDir"],
            f"run_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    if local_rank == 0:
        os.makedirs(args["outputDir"], exist_ok=True)
        with open(os.path.join(args["outputDir"], "args"), "wb") as file:
            pickle.dump(args, file)

    # load data
    with open(args["datasetPath"], "rb") as handle:
        loadedData = pickle.load(handle)

    # data module
    dm = SpeechDataModule(loadedData, args["batchSize"], args["numWorkers"])

    decoder_cfg = None
    decoder_node = args.get("decoder", None)
    if decoder_node is not None:
        decoder_cfg = OmegaConf.to_container(decoder_node, resolve=True)

    lm_decoder_config = None
    fusion_cfg = None
    if decoder_cfg:
        lm_decoder_config = decoder_cfg.get("lm_decoder")
        fusion_cfg = decoder_cfg.get("fusion")

    # tensorboard logger
    logger = TensorBoardLogger(args["outputDir"], name="torch_dist_v0")

    # model
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
        output_dir=args["outputDir"],
        conv_kernel_sizes=list(args["model"]["conv_kernel_sizes"]),
        conv_dilations=list(args["model"]["conv_dilations"]),
        attention_heads=args["model"]["attention_heads"],
        attention_dropout=args["model"].get("attention_dropout", 0.0),
        lm_decoder_config=lm_decoder_config,
    )

    # checkpoint callback
    checkpointCallback = ModelCheckpoint(
        filename=args["outputDir"] + "/modelWeights", monitor="val/ser",
        mode="min", save_top_k=1, every_n_train_steps=None)
    checkpointCallback.FILE_EXTENSION = ""

    callbacks = [checkpointCallback, TimerCallback()]
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

    # trainer
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

    # train
    trainer.fit(model, dm)


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
    if 'wandb' in cfg and cfg.wandb.enabled:
        wandb_config = OmegaConf.to_container(cfg, resolve=True)
        wandb_config['hyperparam_setting'] = conf_name[conf_name.index("_")+1:]
        run_name = f"run_{datetime.now().strftime('%Y%m%d-%H%M%S-')}" + \
            wandb_config['hyperparam_setting']

        print(f"config: {conf_name[conf_name.index('_')+1:]}")
        if local_rank == 0:
            run = wandb.init(**cfg.wandb.setup,
                             config=wandb_config,
                             name=run_name,
                             sync_tensorboard=True)

    trainModel(cfg)
    if 'wandb' in cfg and cfg.wandb.enabled:
        run.finish()


if __name__ == "__main__":
    main()
