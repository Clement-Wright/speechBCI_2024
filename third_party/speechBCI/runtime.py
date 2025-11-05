"""Download helpers for the speechBCI LanguageModelDecoder runtime.

The upstream project (https://github.com/fwillett/speechBCI) distributes
pre-built WFST decoder binaries under the Apache 2.0 license. Shipping those
artefacts inside this repository would significantly increase the repository
size, so we provide a thin wrapper that can fetch the required files on demand.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Optional
from urllib.request import urlopen

LOGGER = logging.getLogger(__name__)

RUNTIME_SUBDIR = Path("runtime") / "server" / "x86"
DEFAULT_REPO_ZIP = "https://github.com/fwillett/speechBCI/archive/refs/heads/main.zip"


def get_runtime_root(base_dir: Optional[Path] = None) -> Path:
    """Return the expected runtime root directory."""

    if base_dir is None:
        base_dir = Path(__file__).resolve().parent
    return base_dir / RUNTIME_SUBDIR


def ensure_runtime(
    *,
    base_dir: Optional[Path] = None,
    download: bool = False,
    repo_zip_url: str = DEFAULT_REPO_ZIP,
) -> Path:
    """Ensure that the runtime directory exists.

    Parameters
    ----------
    base_dir:
        Base directory that contains the `runtime/` folder. Defaults to the
        package directory.
    download:
        When ``True``, attempt to download the runtime artefacts from the
        upstream speechBCI repository if they are missing.
    repo_zip_url:
        URL to a zip archive of the upstream repository. This can be customised
        to point at a fork or a pinned commit.
    """

    runtime_root = get_runtime_root(base_dir)
    if runtime_root.exists():
        return runtime_root

    if not download:
        LOGGER.warning(
            "speechBCI runtime not found at %s. Run download_runtime.py to fetch "
            "the decoder binaries.",
            runtime_root,
        )
        return runtime_root

    LOGGER.info("Downloading speechBCI runtime from %s", repo_zip_url)
    try:
        with urlopen(repo_zip_url) as response:
            data = response.read()
    except Exception as exc:  # pragma: no cover - network failure
        LOGGER.error("Failed to download speechBCI runtime: %s", exc)
        raise

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = Path(tmpdir) / "speechbci.zip"
        with open(zip_path, "wb") as f:
            f.write(data)

        with zipfile.ZipFile(zip_path) as zf:
            members = [
                m
                for m in zf.namelist()
                if "LanguageModelDecoder/runtime/server/x86" in m
            ]
            if not members:
                raise RuntimeError(
                    "Archive did not contain LanguageModelDecoder runtime files"
                )

            zf.extractall(tmpdir)

        # Determine extracted root
        extracted_roots = list(Path(tmpdir).glob("speechBCI-*"))
        if not extracted_roots:
            raise RuntimeError("Unable to locate extracted speechBCI repository")
        extracted_root = extracted_roots[0]
        source_dir = (
            extracted_root / "LanguageModelDecoder" / "runtime" / "server" / "x86"
        )
        if not source_dir.exists():
            raise RuntimeError(
                "Downloaded speechBCI archive missing runtime/server/x86 folder"
            )

        shutil.copytree(source_dir, runtime_root, dirs_exist_ok=True)

    LOGGER.info("speechBCI runtime downloaded to %s", runtime_root)
    return runtime_root


def _main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch speechBCI runtime files")
    parser.add_argument(
        "--repo-zip-url",
        default=DEFAULT_REPO_ZIP,
        help="URL to a zip archive of the speechBCI repository",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even if the runtime directory already exists",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Base directory that contains the runtime folder",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging verbosity",
    )
    args = parser.parse_args(argv)

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    if args.force and get_runtime_root(args.base_dir).exists():
        shutil.rmtree(get_runtime_root(args.base_dir))

    try:
        ensure_runtime(base_dir=args.base_dir, download=True, repo_zip_url=args.repo_zip_url)
    except Exception as exc:  # pragma: no cover - CLI feedback
        LOGGER.error("Failed to download runtime: %s", exc)
        return 1

    LOGGER.info("speechBCI runtime is ready")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
