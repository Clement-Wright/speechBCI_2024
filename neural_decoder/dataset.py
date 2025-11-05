import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

import numpy as np
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


def _normalise_dtype(dtype_like: Optional[Any]) -> np.dtype:
    if dtype_like is None:
        return np.dtype(np.float32)
    return np.dtype(dtype_like)


def _resolve_manifest(base_dir: Path, explicit: Optional[str], candidate_names: Sequence[str]) -> Path:
    candidates: List[Path] = []
    if explicit:
        candidates.append((base_dir / explicit).resolve())
    for name in candidate_names:
        candidates.append((base_dir / name).resolve())
        candidates.append((base_dir / "metadata" / name).resolve())
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Unable to locate a manifest file. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_np_resource(path: Path) -> Any:
    if path.suffix.lower() == ".npy":
        return np.load(path, allow_pickle=True)
    data = np.load(path, allow_pickle=True)
    if len(data.files) == 1:
        return data[data.files[0]]
    return {key: data[key] for key in data.files}


def _load_lines(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as handle:
        return [line.rstrip("\n") for line in handle]


def _coerce_sequence_list(values: Any, dtype: Optional[np.dtype] = None) -> List[np.ndarray]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        if values.dtype == object:
            sequences = [np.asarray(item, dtype=dtype) for item in values]
        elif values.ndim >= 2:
            sequences = [np.asarray(item, dtype=dtype) for item in values]
        else:
            sequences = [np.asarray(values, dtype=dtype)]
    elif isinstance(values, (list, tuple)):
        sequences = [np.asarray(item, dtype=dtype) for item in values]
    else:
        raise TypeError(f"Unsupported sequence container type: {type(values)!r}")
    return [np.asarray(seq, dtype=dtype) for seq in sequences]


def _coerce_int_list(values: Any) -> List[int]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        flat = values.tolist()
    else:
        flat = list(values)
    return [int(item) for item in flat]


def _coerce_string_list(values: Any) -> List[str]:
    if values is None:
        return []
    if isinstance(values, np.ndarray):
        flat = values.tolist()
    else:
        flat = list(values)
    return [str(item) for item in flat]


def _load_day_from_manifest(
    base_dir: Path,
    entry: Dict[str, Any],
    feature_dtype: np.dtype,
) -> Dict[str, Any]:
    fields = entry.get("fields", {})
    if not fields:
        fields = {key: value for key, value in entry.items() if key not in {"session", "split"}}

    def _resolve_field(name: str) -> Optional[Path]:
        value = fields.get(name)
        if value is None:
            return None
        resolved = (base_dir / value).resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Manifest references missing file for field '{name}': {resolved}")
        return resolved

    sentence_path = _resolve_field("sentenceDat") or _resolve_field("features")
    if sentence_path is None:
        raise KeyError("Manifest entry missing 'sentenceDat' (or 'features') reference")

    if sentence_path.suffix.lower() in {".npy", ".npz"}:
        sentence_data = _load_np_resource(sentence_path)
    else:
        raise ValueError(
            f"Unsupported file type for neural features: {sentence_path.suffix}. "
            "Please convert to .npy or .npz."
        )

    phoneme_path = _resolve_field("phonemes")
    phone_len_path = _resolve_field("phoneLens")
    transcription_path = _resolve_field("transcriptions")
    word_dat_path = _resolve_field("wordDat")
    word_len_path = _resolve_field("wordLens")
    words_path = _resolve_field("words")

    phoneme_values = _load_np_resource(phoneme_path) if phoneme_path else []
    phone_len_values = _load_np_resource(phone_len_path) if phone_len_path else []

    if transcription_path:
        if transcription_path.suffix.lower() == ".json":
            transcription_values = _load_json(transcription_path)
        else:
            transcription_values = _load_lines(transcription_path)
    else:
        transcription_values = []

    word_dat_values = _load_np_resource(word_dat_path) if word_dat_path else []
    word_len_values = _load_np_resource(word_len_path) if word_len_path else []
    if words_path:
        if words_path.suffix.lower() == ".json":
            words_values = _load_json(words_path)
        else:
            words_values = _load_lines(words_path)
    else:
        words_values = []

    sentence_sequences = _coerce_sequence_list(sentence_data, dtype=feature_dtype)
    phoneme_sequences = (
        _coerce_sequence_list(phoneme_values, dtype=np.int32) if len(phoneme_values) else []
    )
    if len(phone_len_values):
        phone_lens = _coerce_int_list(phone_len_values)
    elif phoneme_sequences:
        phone_lens = [len(seq) for seq in phoneme_sequences]
    else:
        phone_lens = [0 for _ in sentence_sequences]
    transcriptions = _coerce_string_list(transcription_values)

    if phoneme_sequences and len(phoneme_sequences) != len(sentence_sequences):
        raise ValueError(
            "Mismatch between neural feature sequences and phoneme annotations "
            f"(neural={len(sentence_sequences)}, phonemes={len(phoneme_sequences)})."
        )

    if phoneme_sequences and len(phone_lens) != len(phoneme_sequences):
        raise ValueError(
            "Mismatch between phoneme sequences and phone length annotations "
            f"(phonemes={len(phoneme_sequences)}, phoneLens={len(phone_lens)})."
        )

    day_entry: Dict[str, Any] = {
        "sentenceDat": sentence_sequences,
        "phonemes": phoneme_sequences or [[] for _ in sentence_sequences],
        "phoneLens": phone_lens or [len(seq) for seq in phoneme_sequences],
        "transcriptions": transcriptions or ["" for _ in sentence_sequences],
    }

    if word_dat_values:
        day_entry["wordDat"] = _coerce_sequence_list(word_dat_values, dtype=np.int32)
    if word_len_values:
        day_entry["wordLens"] = _coerce_int_list(word_len_values)
    if words_values:
        day_entry["words"] = _coerce_string_list(words_values)

    return day_entry


def _apply_split_overrides(
    manifest: Dict[str, Any],
    split_overrides: Optional[Dict[str, Iterable[str]]],
) -> Dict[str, List[Dict[str, Any]]]:
    available_splits = manifest.get("splits", {})
    if not available_splits:
        raise KeyError("Manifest file does not contain a 'splits' mapping")

    if split_overrides:
        mapping = {
            target: [name for name in split_names if name in available_splits]
            for target, split_names in split_overrides.items()
        }
    else:
        mapping = {
            "train": [key for key in ("train", "training") if key in available_splits],
            "test": [key for key in ("validation", "val", "dev", "test") if key in available_splits],
        }
        if not mapping["train"]:
            raise ValueError("No training split found in manifest. Provide split_overrides to resolve ambiguity.")
        if not mapping["test"]:
            mapping["test"] = mapping["train"]

    resolved: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    for target_split, split_names in mapping.items():
        seen_sessions: Set[str] = set()
        for split_name in split_names:
            for entry in available_splits.get(split_name, []):
                session_id = entry.get("session") or entry.get("id") or entry.get("name")
                dedupe_key = f"{split_name}:{session_id}"
                if dedupe_key in seen_sessions:
                    continue
                resolved[target_split].append(entry)
                seen_sessions.add(dedupe_key)
    return resolved


def load_dryad2025_dataset(
    dataset_path: Path,
    feature_dtype: np.dtype,
    split_overrides: Optional[Dict[str, Iterable[str]]] = None,
    manifest_file: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    base_path = dataset_path
    if base_path.is_file():
        raise ValueError("Dryad dataset loader expects a directory, got a file path instead")

    if (base_path / "raw").is_dir():
        base_path = base_path / "raw"

    manifest_path = _resolve_manifest(
        base_path,
        manifest_file,
        ("dryad2025_manifest.json", "manifest.json"),
    )
    manifest = _load_json(manifest_path)
    split_map = _apply_split_overrides(manifest, split_overrides)

    result: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    for split, entries in split_map.items():
        for entry in entries:
            result[split].append(_load_day_from_manifest(base_path, entry, feature_dtype))
    return result


def load_kaggle2025_dataset(
    dataset_path: Path,
    feature_dtype: np.dtype,
    split_overrides: Optional[Dict[str, Iterable[str]]] = None,
    manifest_file: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    base_path = dataset_path
    if base_path.is_file():
        raise ValueError("Kaggle dataset loader expects a directory, got a file path instead")

    if (base_path / "raw").is_dir():
        base_path = base_path / "raw"

    manifest_path = _resolve_manifest(
        base_path,
        manifest_file,
        ("kaggle2025_manifest.json", "manifest.json"),
    )
    manifest = _load_json(manifest_path)
    split_map = _apply_split_overrides(manifest, split_overrides)

    result: Dict[str, List[Dict[str, Any]]] = {"train": [], "test": []}
    for split, entries in split_map.items():
        for entry in entries:
            result[split].append(_load_day_from_manifest(base_path, entry, feature_dtype))
    return result


DATASET_LOADERS: Dict[str, Any] = {
    "dryad_2025": load_dryad2025_dataset,
    "dryad2025": load_dryad2025_dataset,
    "kaggle_2025": load_kaggle2025_dataset,
    "kaggle2025": load_kaggle2025_dataset,
}


def load_dataset(
    dataset_path: str,
    dataset_type: Optional[str],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    path = Path(dataset_path)

    if path.is_file() and (dataset_type is None or dataset_type == "pickle"):
        with path.open("rb") as handle:
            return pickle.load(handle)

    if dataset_type is None:
        raise ValueError("dataset_type must be provided when loading from a directory")

    loader_key = dataset_type.lower()
    if loader_key not in DATASET_LOADERS:
        raise KeyError(f"Unknown dataset type: {dataset_type}")

    loader_options = dict(options or {})

    feature_dtype = _normalise_dtype(loader_options.pop("feature_dtype", None))

    split_overrides = loader_options.pop("split_overrides", None)
    train_split = loader_options.pop("train_split", None)
    eval_split = loader_options.pop("eval_split", None)
    if split_overrides is None:
        overrides: Dict[str, Iterable[str]] = {}
        if train_split is not None:
            overrides["train"] = (
                [train_split] if isinstance(train_split, str) else list(train_split)
            )
        if eval_split is not None:
            overrides["test"] = (
                [eval_split] if isinstance(eval_split, str) else list(eval_split)
            )
        split_overrides = overrides or None

    loader = DATASET_LOADERS[loader_key]
    return loader(
        path,
        feature_dtype=feature_dtype,
        split_overrides=split_overrides,
        **loader_options,
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset materialisation helper")
    parser.add_argument("--dataset-path", required=True, help="Path to the dataset directory")
    parser.add_argument("--dataset-type", required=True, choices=sorted(DATASET_LOADERS), help="Dataset type key")
    parser.add_argument("--output", required=True, help="Destination path for the pickle file")
    parser.add_argument(
        "--feature-dtype",
        default="float32",
        help="NumPy dtype for neural features (default: %(default)s)",
    )
    parser.add_argument("--train-split", help="Override manifest split for training data")
    parser.add_argument("--eval-split", help="Override manifest split for evaluation data")
    parser.add_argument(
        "--manifest",
        help="Explicit manifest filename relative to the dataset root",
    )
    return parser


def _main_cli() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    options: Dict[str, Any] = {"feature_dtype": args.feature_dtype}
    if args.train_split:
        options["train_split"] = args.train_split
    if args.eval_split:
        options["eval_split"] = args.eval_split
    if args.manifest:
        options["manifest_file"] = args.manifest

    dataset = load_dataset(args.dataset_path, args.dataset_type, options=options)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        pickle.dump(dataset, handle)
    print(f"Wrote dataset pickle to {output_path}")


if __name__ == "__main__":
    _main_cli()
