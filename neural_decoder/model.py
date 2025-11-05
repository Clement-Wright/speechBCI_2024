from .augmentations import GaussianSmoothing
from edit_distance import SequenceMatcher
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torch.nn.functional as F


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
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(
                hidden_dim, n_classes + 1)  # +1 for CTC blank

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
        seq_out = self.fc_decoder_out(hid)
        return seq_out

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
