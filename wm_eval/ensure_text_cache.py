#!/usr/bin/env python3
"""Preencode fixed-state RoboTwin prompts missing from the Wan T5 cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from benchmarks.robotwin.prompt_template import format_prompt_for_inference
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.loader import load_wan_components
from openwam.model.video_backbone.wan.pipeline_builder import (
    _filter_text_encoder_configs,
    discover_model_files,
)
from openwam.model.video_backbone.wan.text_embedding_cache import TextEmbeddingCache


def text_encoder_components(model_dir: str, device: str):
    model_configs, tokenizer_config = discover_model_files(model_dir)
    other_configs = _filter_text_encoder_configs(model_configs)
    other_ids = {str(config.path) for config in other_configs}
    text_configs = [config for config in model_configs if str(config.path) not in other_ids]
    if not text_configs:
        raise RuntimeError(f"No Wan T5 weights found in {model_dir}")
    holder = load_wan_components(
        text_configs, tokenizer_config, device=device, torch_dtype=torch.bfloat16
    )
    if holder.text_encoder is None or holder.tokenizer is None:
        raise RuntimeError("Wan T5 encoder or tokenizer did not load")
    holder.text_encoder.eval()
    return holder.text_encoder, holder.tokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    specs = []
    for checkpoint_dir in args.checkpoint_dir:
        with (checkpoint_dir / "config.yaml").open(encoding="utf-8") as handle:
            video = yaml.safe_load(handle)["model"]["video_backbone"]
        specs.append((str(video["text_embedding_cache_dir"]), str(video["model_path"])))
    if len(set(specs)) != 1:
        raise ValueError(f"Checkpoints use different text encoders or caches: {specs}")
    cache_dir, model_dir = specs[0]
    cache = TextEmbeddingCache(cache_dir)
    if cache.hash_identity.get("model_path") != model_dir:
        raise ValueError("Text cache model_path differs from checkpoint model_path")

    prompts = []
    for manifest_path in args.manifest:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        prompts.extend(format_prompt_for_inference(str(state["instruction"])) for state in manifest["states"])
    missing = [prompt for prompt in dict.fromkeys(prompts) if not cache.path_for_prompt(prompt).is_file()]
    print(f"[text-cache] prompts={len(set(prompts))} missing={len(missing)} cache={cache_dir}", flush=True)
    if not missing:
        return

    text_encoder, tokenizer = text_encoder_components(model_dir, args.device)
    for prompt in missing:
        with torch.no_grad():
            context, seq_lens = wan_encode.encode_text(
                [prompt], tokenizer=tokenizer, text_encoder=text_encoder, device=torch.device(args.device)
            )
        seq_len = int(seq_lens[0].item())
        path = cache.path_for_prompt(prompt)
        record = {
            "context": context[0, :seq_len].detach().cpu().to(torch.bfloat16).contiguous(),
            "seq_len": seq_len,
            "prompt": prompt,
            "meta": {
                "cache_version": 1,
                "hash_identity": cache.hash_identity,
                "model_path": model_dir,
                "storage": "cropped_to_seq_len",
            },
        }
        temporary = path.with_suffix(".tmp")
        torch.save(record, temporary)
        temporary.replace(path)
        print(f"[text-cache] encoded {path.name}", flush=True)


if __name__ == "__main__":
    main()
