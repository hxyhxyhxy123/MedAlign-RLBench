from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import yaml


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/deploy_kvcache.yaml")
    parser.add_argument("--out", default="runs/kvcache_benchmark.json")
    parser.add_argument("--backend", choices=["auto", "vllm", "transformers"], default="vllm")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    prompts = cfg.get("benchmark", {}).get("prompts", [])
    concurrencies = cfg.get("benchmark", {}).get("concurrency", [1, 4, 8])
    max_new_tokens = int(cfg.get("benchmark", {}).get("max_new_tokens", 256))
    print("KV cache / vLLM PagedAttention benchmark config loaded.")
    print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    if args.dry_run:
        return

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available. Run this after GPU boot and LoRA merge.")

    backend = args.backend
    if backend == "auto":
        try:
            import vllm  # noqa: F401

            backend = "vllm"
        except Exception:
            backend = "transformers"

    if backend == "transformers":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(cfg["model"]), trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            str(cfg["model"]),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        def generate_batch(batch_prompts: list[str], use_cache: bool, tokens: int) -> dict[str, Any]:
            encoded = tokenizer(batch_prompts, return_tensors="pt", padding=True)
            encoded = {k: v.to("cuda") for k, v in encoded.items()}
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                outputs = model.generate(
                    **encoded,
                    max_new_tokens=tokens,
                    do_sample=False,
                    use_cache=use_cache,
                    pad_token_id=tokenizer.eos_token_id,
                )
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            out_tokens = sum(max(0, len(row) - encoded["input_ids"].shape[1]) for row in outputs)
            return {
                "seconds": elapsed,
                "output_tokens": int(out_tokens),
                "decode_tokens_per_second": out_tokens / elapsed if elapsed > 0 else 0.0,
                "avg_latency_seconds": elapsed / len(batch_prompts) if batch_prompts else 0.0,
            }

        torch.cuda.reset_peak_memory_stats()
        results = []
        for concurrency in concurrencies:
            repeat = math.ceil(int(concurrency) / max(1, len(prompts)))
            batch_prompts = (prompts * repeat)[: int(concurrency)]
            cached = generate_batch(batch_prompts, use_cache=True, tokens=max_new_tokens)
            cached.update({"concurrency": int(concurrency), "requests": len(batch_prompts), "use_cache": True})
            results.append(cached)

        no_cache_tokens = min(64, max_new_tokens)
        no_cache = generate_batch(prompts[:1], use_cache=False, tokens=no_cache_tokens)
        no_cache.update({"concurrency": 1, "requests": 1, "use_cache": False, "max_new_tokens": no_cache_tokens})
        report = {
            "engine": "transformers",
            "attention": "native generation with use_cache true/false KV-cache comparison",
            "model": cfg["model"],
            "max_new_tokens": max_new_tokens,
            "peak_torch_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "results": results,
            "no_cache_baseline": no_cache,
        }
        write_json(Path(args.out), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from vllm import LLM, SamplingParams

    torch.cuda.reset_peak_memory_stats()
    llm = LLM(
        model=str(cfg["model"]),
        trust_remote_code=True,
        dtype=cfg.get("dtype", "bfloat16"),
        gpu_memory_utilization=float(cfg.get("gpu_memory_utilization", 0.85)),
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
    )

    results = []
    for concurrency in concurrencies:
        repeat = math.ceil(int(concurrency) / max(1, len(prompts)))
        batch_prompts = (prompts * repeat)[: int(concurrency)]
        started = time.perf_counter()
        outputs = llm.generate(batch_prompts, sampling)
        elapsed = time.perf_counter() - started
        out_tokens = sum(len(item.outputs[0].token_ids) for item in outputs)
        results.append(
            {
                "concurrency": int(concurrency),
                "requests": len(batch_prompts),
                "seconds": elapsed,
                "output_tokens": out_tokens,
                "decode_tokens_per_second": out_tokens / elapsed if elapsed > 0 else 0.0,
                "avg_latency_seconds": elapsed / len(batch_prompts) if batch_prompts else 0.0,
            }
        )

    report = {
        "engine": "vllm",
        "attention": "PagedAttention with KV cache",
        "model": cfg["model"],
        "max_new_tokens": max_new_tokens,
        "peak_torch_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "results": results,
    }
    write_json(Path(args.out), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
