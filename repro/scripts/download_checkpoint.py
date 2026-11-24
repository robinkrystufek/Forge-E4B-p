"""Resumably download the single pinned QAT weights file outside Git."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from inspect_checkpoint_header import DEFAULT_REPO, DEFAULT_REVISION

MODEL_BYTES = 3_525_094_516
MODEL_ETAG = "d560c03a1a65d472df245005745177aa886dd7c7848accc4d582b73cfe0a0df9"


def download(repo: str, revision: str, output: Path, metadata_path: Path, retries: int = 4) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    url = f"https://huggingface.co/{repo}/resolve/{revision}/model.safetensors"
    attempts = 0
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == MODEL_BYTES:
            break
        if offset > MODEL_BYTES:
            raise RuntimeError(f"partial file exceeds expected checkpoint size: {offset} > {MODEL_BYTES}")
        try:
            headers = {"User-Agent": "Forge-GM-mobile-QAT-feasibility/0.1"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                if response.headers.get("ETag", "").strip('"') != MODEL_ETAG:
                    raise RuntimeError("checkpoint ETag differs from inspected artifact")
                if offset and (
                    response.status != 206
                    or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")
                ):
                    raise RuntimeError("server did not honor resume range; refusing duplicate append")
                with partial.open("ab" if offset else "wb") as stream:
                    while chunk := response.read(8 * 1024 * 1024):
                        stream.write(chunk)
                        stream.flush()
            if partial.stat().st_size != MODEL_BYTES:
                raise RuntimeError(f"length mismatch: {partial.stat().st_size} != {MODEL_BYTES}")
            break
        except (OSError, urllib.error.URLError, TimeoutError) as exc:
            attempts += 1
            if attempts >= retries:
                raise
            print(f"recoverable download error ({attempts}/{retries}): {exc}", flush=True)
            time.sleep(min(60, 2**attempts))

    digest = hashlib.sha256()
    with partial.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    partial.replace(output)
    metadata = {
        "repo_id": repo,
        "revision": revision,
        "filename": output.name,
        "size_bytes": output.stat().st_size,
        "sha256": sha256,
        "hub_etag": MODEL_ETAG,
        "path": str(output.resolve()),
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    tmp.replace(metadata_path)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--retries", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(download(args.repo, args.revision, args.output, args.metadata, args.retries), indent=2))


if __name__ == "__main__":
    main()
