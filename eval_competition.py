import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
try:
    from edit_distance import SequenceMatcher
except ImportError:  # pragma: no cover - optional when only formatting submissions
    SequenceMatcher = None

try:
    import torch
except ImportError:  # pragma: no cover - allows helper import without torch
    torch = None

try:
    from jiwer import wer
except ImportError:  # pragma: no cover - optional dependency for metrics
    wer = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

if TYPE_CHECKING:  # pragma: no cover - hints for type checkers only
    from neural_decoder.dataset import SpeechDataset
    from neural_decoder_trainer import loadModel
    import NeuralDecoder.neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils


def _prepare_adapter(model, args) -> None:
    adapter = getattr(model, "decoder_adapter", None)
    if adapter is None:
        raise RuntimeError("Loaded model does not expose a language model adapter")

    resource_overrides = {
        key: value
        for key, value in {
            "fst_path": args.fst_path,
            "const_arpa_path": args.const_arpa_path,
            "g_path": args.g_path,
            "words_path": args.words_path,
            "symbol_table": args.symbol_table,
        }.items()
        if value
    }

    decode_options: Dict[str, float] = {}
    if args.beam_width is not None:
        decode_options["nbest"] = args.beam_width

    if resource_overrides or decode_options or args.logit_mapping is not None:
        adapter.configure_backend(
            resource=resource_overrides or None,
            decode_options=decode_options or None,
            logit_mapping=args.logit_mapping,
        )

    if not adapter.backend_available:
        raise RuntimeError(
            "Language model backend is unavailable. Ensure the decoder runtime "
            "is built and the resource paths are correctly specified."
        )


def _prepare_submission_records(
    predictions: Sequence[str],
    sample_ids: Optional[Iterable[int]] = None,
) -> List[Dict[str, str]]:
    if sample_ids is None:
        sample_ids = range(len(predictions))
    return [
        {"id": int(sample_id), "text": prediction}
        for sample_id, prediction in zip(sample_ids, predictions)
    ]


def save_submission(
    predictions: Sequence[str],
    output_dir: str,
    submission_format: str = "csv",
    filename: Optional[str] = None,
    compress: bool = False,
    sample_ids: Optional[Iterable[int]] = None,
) -> Tuple[Path, Optional[Path]]:
    """Write predictions to disk in the Kaggle submission format.

    Args:
        predictions: Ordered predictions as decoded strings.
        output_dir: Directory where the submission file will be written.
        submission_format: Either ``"csv"`` or ``"json"``.
        filename: Optional custom filename. Extension is inferred from
            ``submission_format`` if not provided.
        compress: If ``True``, create a ``.zip`` archive with the submission file.
        sample_ids: Optional iterable of integer identifiers. Defaults to
            ``range(len(predictions))``.

    Returns:
        A tuple containing the path to the generated submission file and the path
        to the optional zip archive (``None`` if ``compress`` is ``False``).
    """

    if submission_format not in {"csv", "json"}:
        raise ValueError("submission_format must be 'csv' or 'json'")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inferred_name = f"submission.{submission_format}"
    submission_name = filename or inferred_name
    submission_path = out_dir / submission_name

    records = _prepare_submission_records(predictions, sample_ids)

    if submission_format == "csv":
        with submission_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id", "text"])
            for record in records:
                writer.writerow([record["id"], record["text"]])
    else:
        with submission_path.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, ensure_ascii=False, indent=2)

    zip_path: Optional[Path] = None
    if compress:
        zip_name = submission_path.with_suffix(submission_path.suffix + ".zip")
        zip_path = zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(submission_path, arcname=submission_path.name)

    return submission_path, zip_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate decoder with optional LM fusion")
    parser.add_argument("--modelPath", type=str, required=True, help="Path to model weights")
    parser.add_argument(
        "--dataPath",
        type=str,
        default="/home/<user>/speech_BCI_torch/data/ptDecoder_ctc",
        help="Path to parsed dataset",
    )
    parser.add_argument(
        "--MODEL_CACHE_DIR",
        type=str,
        default="/home/<user>/speech_BCI/data/LLM/opt_model/",
        help="Path to LLM cache (baseline mode)",
    )
    parser.add_argument(
        "--lmDir",
        type=str,
        default="/home/<user>/speech_BCI/data/speech_5gram/lang_test",
        help="Path to language model directory (baseline mode)",
    )
    parser.add_argument("--outputDir", type=str, default="./eval_output", help="Output directory")
    parser.add_argument(
        "--decoder-mode",
        choices=["baseline", "language_model"],
        default="baseline",
        help="Select the decoding strategy",
    )
    parser.add_argument("--beam-width", type=int, default=None, help="Beam width for WFST decoding")
    parser.add_argument("--temperature", type=float, default=1.0, help="Temperature for WFST decoding")
    parser.add_argument("--fst-path", type=str, default=None, help="Path to TLG.fst")
    parser.add_argument("--const-arpa-path", type=str, default=None, help="Path to const arpa model")
    parser.add_argument("--g-path", type=str, default=None, help="Path to G.fst")
    parser.add_argument("--words-path", type=str, default=None, help="Path to words.txt")
    parser.add_argument("--symbol-table", type=str, default=None, help="Path to symbol table")
    parser.add_argument(
        "--logit-mapping",
        type=int,
        nargs="+",
        default=None,
        help="Optional permutation applied to logits before decoding",
    )
    parser.add_argument(
        "--submission-format",
        choices=["csv", "json"],
        default="csv",
        help="Submission file format required by the Kaggle competition",
    )
    parser.add_argument(
        "--submission-name",
        type=str,
        default=None,
        help="Optional custom name for the submission file",
    )
    parser.add_argument(
        "--compress-submission",
        action="store_true",
        help="Create a ZIP archive alongside the raw submission file",
    )

    args = parser.parse_args()

    if torch is None:
        raise ImportError(
            "PyTorch is required to run eval_competition. Install torch before executing the evaluator."
        )

    from neural_decoder.dataset import SpeechDataset
    from neural_decoder_trainer import loadModel
    import NeuralDecoder.neuralDecoder.utils.lmDecoderUtils as lmDecoderUtils

    with open(args.dataPath, "rb") as handle:
        loadedData = pickle.load(handle)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = loadModel(args.modelPath, device=device)
    model.eval()

    if args.decoder_mode == "language_model":
        _prepare_adapter(model, args)

    rnn_outputs: Dict[str, List] = {
        "logits": [],
        "logitLengths": [],
        "trueSeqs": [],
        "transcriptions": [],
        "transcriptions_raw": [],
    }
    lm_hypotheses: List[List[Dict]] = []

    partition = "competition"
    evaluation_days = [4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20]
    for i, testDayIdx in enumerate(evaluation_days):
        test_ds = SpeechDataset([loadedData[partition][i]])
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=1, shuffle=False, num_workers=0
        )
        for j, (X, y, X_len, y_len, _) in enumerate(test_loader):
            X, y, X_len, y_len, dayIdx = (
                X.to(device),
                y.to(device),
                X_len.to(device),
                y_len.to(device),
                torch.tensor([testDayIdx], dtype=torch.int64).to(device),
            )

            with torch.no_grad():
                pred = model.forward(X, dayIdx)
            adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(torch.int32)

            for iterIdx in range(pred.shape[0]):
                trueSeq = np.array(y[iterIdx][0: y_len[iterIdx]].cpu().detach())
                rnn_outputs["logits"].append(pred[iterIdx].cpu().detach().numpy())
                rnn_outputs["logitLengths"].append(
                    adjustedLens[iterIdx].cpu().detach().item()
                )
                rnn_outputs["trueSeqs"].append(trueSeq)

            transcript = loadedData[partition][i]["transcriptions"][j].strip()
            transcript = re.sub(r"[^a-zA-Z\- \']", "", transcript)
            transcript = transcript.replace("--", "").lower()
            rnn_outputs["transcriptions"].append(transcript)
            rnn_outputs["transcriptions_raw"].append(transcript)

            if args.decoder_mode == "language_model":
                batch_hyps = model.decode_with_language_model(
                    pred.detach(),
                    adjustedLens,
                    beam_width=args.beam_width,
                    temperature=args.temperature,
                )
                lm_hypotheses.extend(batch_hyps)

    os.makedirs(args.outputDir, exist_ok=True)

    if args.decoder_mode == "baseline":
        llm, llm_tokenizer = lmDecoderUtils.build_opt(
            cacheDir=args.MODEL_CACHE_DIR, device="auto", load_in_8bit=True
        )
        ngramDecoder = lmDecoderUtils.build_lm_decoder(
            args.lmDir, acoustic_scale=0.5, nbest=100, beam=18
        )

        acoustic_scale = 0.5
        blank_penalty = np.log(7)
        llm_weight = 0.5

        start_t = time.time()
        nbest_outputs = []
        for logits in rnn_outputs["logits"]:
            logits = np.concatenate([logits[:, 1:], logits[:, 0:1]], axis=-1)
            logits = lmDecoderUtils.rearrange_speech_logits(
                logits[None, :, :], has_sil=True
            )
            nbest = lmDecoderUtils.lm_decode(
                ngramDecoder,
                logits[0],
                blankPenalty=blank_penalty,
                returnNBest=True,
                rescore=True,
            )
            nbest_outputs.append(nbest)
        time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
        print(f"5gram decoding took {time_per_sample} seconds per sample")

        for i in range(len(rnn_outputs["transcriptions"])):
            new_trans = [ord(c) for c in rnn_outputs["transcriptions"][i]] + [0]
            rnn_outputs["transcriptions"][i] = np.array(new_trans)

        start_t = time.time()
        llm_out = lmDecoderUtils.cer_with_gpt2_decoder(
            llm,
            llm_tokenizer,
            nbest_outputs[:],
            acoustic_scale,
            rnn_outputs,
            outputType="speech_sil",
            returnCI=True,
            lengthPenalty=0,
            alpha=llm_weight,
        )
        llm_time_per_sample = (time.time() - start_t) / len(rnn_outputs["logits"])
        print(f"LLM decoding took {llm_time_per_sample} seconds per sample")
        print(llm_out["cer"], llm_out["wer"])

        with open(os.path.join(args.outputDir, "llm_out"), "wb") as handle:
            pickle.dump(llm_out, handle)

        decodedTranscriptions = llm_out["decoded_transcripts"]
        submission_name = "5gramLLMCompetitionSubmission.txt"
        with open(os.path.join(args.outputDir, submission_name), "w") as f:
            for transcription in decodedTranscriptions:
                f.write(transcription + "\n")
    else:
        decodedTranscriptions = []
        for hyp in lm_hypotheses:
            if hyp:
                decodedTranscriptions.append(hyp[0].get("sentence", ""))
            else:
                decodedTranscriptions.append("")

        if SequenceMatcher is None:
            print("Install 'edit-distance' to compute CER metrics.")
        else:
            # Compute a simple character error rate against the reference text
            total_distance = 0.0
            total_length = 0
            for predicted, reference in zip(
                decodedTranscriptions, rnn_outputs["transcriptions_raw"]
            ):
                matcher = SequenceMatcher(a=list(reference), b=list(predicted))
                total_distance += matcher.distance()
                total_length += len(reference)

            if total_length > 0:
                cer = total_distance / total_length
                print(f"WFST decoder char error rate: {cer:.4f}")

        if wer is not None:
            dataset_wer = wer(rnn_outputs["transcriptions_raw"], decodedTranscriptions)
            print(f"WFST decoder word error rate: {dataset_wer:.4f}")
        else:
            print("Install 'jiwer' to compute WER metrics.")

        submission_name = "LanguageModelCompetitionSubmission.txt"
        with open(os.path.join(args.outputDir, submission_name), "w") as f:
            for transcription in decodedTranscriptions:
                f.write(transcription + "\n")

    submission_path, zip_path = save_submission(
        decodedTranscriptions,
        args.outputDir,
        submission_format=args.submission_format,
        filename=args.submission_name,
        compress=args.compress_submission,
    )
    print(f"Saved Kaggle submission file to {submission_path}")
    if zip_path is not None:
        print(f"Saved compressed submission archive to {zip_path}")


if __name__ == "__main__":
    main()
