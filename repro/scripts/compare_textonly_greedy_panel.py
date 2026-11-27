"""Compare official and text-only greedy outputs on the frozen 12-case panel.

The installed LiteRT-LM Session API returns decoded text, not native generated
token IDs. This script records decoded output hashes and first text divergence;
it does not claim these are generated-token-ID or logits comparisons.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import litert_lm
import sentencepiece


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_case(engine: litert_lm.Engine, tokenizer, prompt: str, max_output: int) -> dict:
    session = engine.create_session(
        apply_prompt_template=False,
        sampler_config=litert_lm.SamplerConfig(top_k=1, top_p=1.0, temperature=0.0),
        max_output_tokens=max_output,
        enable_speculative_decoding=False,
    )
    try:
        start = time.perf_counter()
        session.run_prefill([prompt])
        response = session.run_decode()
        elapsed = time.perf_counter() - start
        text = response.texts[0] if response.texts else ""
        retokenized_ids = tokenizer.encode(text, out_type=int)
        id_bytes = b"".join(int(token).to_bytes(4, "little", signed=False) for token in retokenized_ids)
        return {"text": text, "sha256": sha(text), "elapsed_seconds": elapsed,
                "decoded_characters": len(text), "retokenized_output_ids_sha256": hashlib.sha256(id_bytes).hexdigest(),
                "retokenized_output_token_count": len(retokenized_ids), "_retokenized_ids": retokenized_ids}
    finally:
        session.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--official", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--panel", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument("--max-output", type=int, default=128)
    args = p.parse_args()
    dll_dir = os.add_dll_directory(str(Path(litert_lm.__file__).resolve().parent))
    backend = litert_lm.Backend.GPU() if args.backend == "gpu" else litert_lm.Backend.CPU()
    panel = json.loads(args.panel.read_text(encoding="utf-8"))["cases"]
    if len(panel) != 12:
        raise ValueError(f"Expected exactly 12 frozen cases, found {len(panel)}")
    cache = args.cache.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(args.tokenizer.resolve()))
    results = []
    for model_name, model_path in (("official", args.official), ("text-only", args.candidate)):
        with model_path.open("rb") as f:
            package_sha = hashlib.file_digest(f, "sha256").hexdigest()
        model_cache = cache / model_name
        model_cache.mkdir(parents=True, exist_ok=True)
        engine = litert_lm.Engine(str(model_path.resolve()), backend=backend,
                                  max_num_tokens=4096, cache_dir=str(model_cache),
                                  enable_benchmark=True)
        try:
            for case in panel:
                prompt = case["serialized_prompt"]
                result = compare_case(engine, tokenizer, prompt, args.max_output)
                results.append({"case_id": case["case_id"], "prompt_sha256": case["prompt_sha256"],
                                "package": model_name, "package_sha256": package_sha, **result})
                print(json.dumps({k: v for k, v in results[-1].items() if k not in ("text", "_retokenized_ids")}), flush=True)
        finally:
            engine.close()
    by_case: dict[str, dict[str, dict]] = {}
    for row in results:
        by_case.setdefault(row["case_id"], {})[row["package"]] = row
    pairs = []
    for case_id, pair in by_case.items():
        a, b = pair["official"]["text"], pair["text-only"]["text"]
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        ids_a, ids_b = pair["official"]["_retokenized_ids"], pair["text-only"]["_retokenized_ids"]
        first_id = next((i for i, (x, y) in enumerate(zip(ids_a, ids_b)) if x != y), min(len(ids_a), len(ids_b)))
        pairs.append({"case_id": case_id, "official_sha256": pair["official"]["sha256"],
                      "text_only_sha256": pair["text-only"]["sha256"],
                      "decoded_text_equal": a == b, "first_divergent_character": None if a == b else first,
                      "retokenized_token_ids_equal": ids_a == ids_b,
                      "first_divergent_retokenized_token_id": None if ids_a == ids_b else first_id,
                      "official_retokenized_token_count": len(ids_a),
                      "text_only_retokenized_token_count": len(ids_b),
                      "official_decoded_characters": len(a), "text_only_decoded_characters": len(b)})
    output = {"schema": "textonly-greedy-panel-v1", "backend": args.backend,
              "runtime": "LiteRT-LM Python Session API 0.19.2", "sampler": {
                  "top_k": 1, "top_p": 1.0, "temperature": 0.0, "mtp": "off"},
              "max_context_tokens": 4096, "max_output_tokens": args.max_output,
              "panel_case_count": len(panel), "native_generated_token_ids_available": False,
              "logits_available": False,
              "comparison_note": "Native generated token IDs and per-step logits are not exposed; the report also compares SentencePiece token IDs obtained by re-tokenizing decoded output, which are not claimed to be the original sampled ID stream.",
              "pairs": pairs, "runs": [{k: v for k, v in row.items() if k not in ("text", "_retokenized_ids")} for row in results]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    _ = dll_dir


if __name__ == "__main__":
    main()
