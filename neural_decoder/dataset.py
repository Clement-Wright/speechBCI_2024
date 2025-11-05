import random
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


class SpeechDataset(Dataset):
    def __init__(self, data, transform=None, options: Optional[Dict] = None):
        self.transform = transform
        self.options = dict(options or {})

        allowed_tokens = self.options.get("vocabulary") or self.options.get("allowed_tokens")
        if allowed_tokens is not None:
            allowed_tokens = {int(tok) for tok in allowed_tokens}
            allowed_tokens.add(0)

        self.max_time_bins = self.options.get("max_time_bins")
        self.max_phone_len = self.options.get("max_phone_len", self.options.get("max_seq_len"))
        self.random_crop = bool(self.options.get("random_crop", False))

        self.entries = []
        for day, day_data in enumerate(data):
            sentences = day_data["sentenceDat"]
            phonemes = day_data["phonemes"]
            phone_lens = day_data["phoneLens"]

            for trial_idx, sentence in enumerate(sentences):
                phone_len = int(phone_lens[trial_idx])
                phone_seq = phonemes[trial_idx][:phone_len]

                if allowed_tokens is not None and any(
                    int(token) not in allowed_tokens for token in phone_seq
                ):
                    continue

                self.entries.append(
                    {
                        "neural": sentence,
                        "phonemes": phonemes[trial_idx],
                        "phone_len": phone_len,
                        "day": day,
                    }
                )

        if not self.entries:
            raise ValueError("No samples available after applying dataset filters.")

    def __len__(self):
        return len(self.entries)

    def _crop_sequence(self, tensor: torch.Tensor, max_length: Optional[int]) -> torch.Tensor:
        if max_length is None or tensor.shape[0] <= max_length:
            return tensor

        if self.random_crop and tensor.shape[0] > max_length:
            start = random.randint(0, tensor.shape[0] - max_length)
        else:
            start = 0
        return tensor[start : start + max_length]

    def __getitem__(self, idx):
        entry = self.entries[idx]

        neural_feats = torch.tensor(entry["neural"], dtype=torch.float32)
        neural_feats = self._crop_sequence(neural_feats, self.max_time_bins)

        phone_seq = torch.tensor(
            entry["phonemes"][: entry["phone_len"]], dtype=torch.int32
        )
        phone_seq = self._crop_sequence(phone_seq, self.max_phone_len)

        if self.transform:
            neural_feats = self.transform(neural_feats)

        return (
            neural_feats,
            phone_seq,
            torch.tensor(neural_feats.shape[0], dtype=torch.int32),
            torch.tensor(phone_seq.shape[0], dtype=torch.int32),
            torch.tensor(entry["day"], dtype=torch.int64),
        )


class SpeechDataModule(pl.LightningDataModule):
    def __init__(
        self,
        loaded_data,
        batch_size,
        num_workers,
        train_transform=None,
        eval_transform=None,
        dataset_options: Optional[Dict] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.loaded_data = loaded_data
        self.train_transform = train_transform
        self.eval_transform = eval_transform
        self.dataset_options = dict(dataset_options or {})

    def setup(self, stage):
        train_options = self.dataset_options.get("train", self.dataset_options)
        eval_options = self.dataset_options.get("eval", self.dataset_options)

        if isinstance(train_options, Dict):
            train_options = dict(train_options)
        if isinstance(eval_options, Dict):
            eval_options = dict(eval_options)

        self.train_ds = SpeechDataset(
            self.loaded_data["train"],
            transform=self.train_transform,
            options=train_options,
        )
        self.test_ds = SpeechDataset(
            self.loaded_data["test"],
            transform=self.eval_transform,
            options=eval_options,
        )

    def _padding(self, batch):
        X, y, X_lens, y_lens, days = zip(*batch)
        X_padded = pad_sequence(X, batch_first=True, padding_value=0)
        y_padded = pad_sequence(y, batch_first=True, padding_value=0)

        return (
            X_padded,
            y_padded,
            torch.stack(X_lens),
            torch.stack(y_lens),
            torch.stack(days),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=self._padding,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_ds,
            num_workers=self.num_workers,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=self._padding,
        )
