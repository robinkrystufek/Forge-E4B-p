"""Fetch the tokenizer/config sidecars into the task-local model directory."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from inspect_checkpoint_header import DEFAULT_REPO, DEFAULT_REVISION

FILES = (
    "README.md", "chat_template.jinja", "config.json", "generation_config.json",
    "preprocessor_config.json", "processor_config.json", "tokenizer.json", "tokenizer_config.json",
)


def fetch(repo: str, revision: str, output_dir: Path) -> list[dict[str, str | int]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
        last_error = None
        for attempt in range(4):
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "Forge-GM-mobile-QAT-feasibility/0.1"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = response.read()
                path = output_dir / name
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_bytes(payload)
                tmp.replace(path)
                records.append({"filename": name, "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()})
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                if attempt < 3:
                    time.sleep(2**attempt)
        if last_error is not None:
            raise last_error
    manifest = {"repo_id": repo, "revision": revision, "files": records}
    path = output_dir / "metadata_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fetch(args.repo, args.revision, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
