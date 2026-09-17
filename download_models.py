"""Pobiera komponenty diffusers modelu MiniMax-Music3 do cache HF (HF_HOME) podczas `docker build`.

Repo HF MiniMaxAI/MiniMax-Music3 ma ~57 GB, ale pipeline diffusers (modular_model_index.json) używa
tylko ~28,5 GB: language_model/ (Qwen3-8B, 17,2 GB w 4 shardach), transformer/ (flow matching 2.4B,
9,7 GB w 2 shardach), rvq_depth_decoder/ (1,3 GB), condition_encoder/, vocoder/, tokenizer/, scheduler/.
Pomijamy qwen_7B/ (wagi) i pliki *.pth (ścieżka SGLang-Omni) oraz assets/figures.

Pobieranie jest podzielone na części = osobne warstwy Dockera, każda < 5 GB (limit warstwy w GHCR
to 10 GB; mniejsze warstwy RunPod pobiera równolegle). Rozmiary (2026-09-17):
    lm1 4,91  lm2 4,92  lm3 4,98  lm4 2,36  fm1 4,96  fm2 4,77  rest 1,64  GB
    python download_models.py lm1|lm2|lm3|lm4|fm1|fm2|rest|all
"""
import os
import sys

from huggingface_hub import snapshot_download

REPO = os.environ.get("MM3_MODEL_ID", "MiniMaxAI/MiniMax-Music3")
PARTS = {
    "lm1": ["language_model/model-00001-of-00004.safetensors"],
    "lm2": ["language_model/model-00002-of-00004.safetensors"],
    "lm3": ["language_model/model-00003-of-00004.safetensors"],
    "lm4": ["language_model/model-00004-of-00004.safetensors", "language_model/*.json"],
    "fm1": ["transformer/diffusion_pytorch_model-00001-of-00002.safetensors"],
    "fm2": ["transformer/diffusion_pytorch_model-00002-of-00002.safetensors", "transformer/*.json"],
    "rest": [
        "rvq_depth_decoder/*",
        "condition_encoder/*",
        "vocoder/*",
        "tokenizer/*",
        "scheduler/*",
        "*.json",  # config.json, modular_model_index.json (+ małe json z qwen_7B/, nieszkodliwe)
        "LICENSE*",
    ],
}


def main() -> None:
    part = (sys.argv[1] if len(sys.argv) > 1 else "all").strip().lower()
    names = list(PARTS) if part == "all" else [part]
    token = os.environ.get("HF_TOKEN") or None
    for name in names:
        patterns = PARTS.get(name)
        if not patterns:
            print(f"[bake] nieznana część: {name} (dostępne: {', '.join(PARTS)})", file=sys.stderr)
            sys.exit(2)
        print(f"[bake] {REPO} część {name}: {patterns} -> HF_HOME={os.environ.get('HF_HOME', '~/.cache/huggingface')}")
        path = snapshot_download(repo_id=REPO, allow_patterns=patterns, token=token, max_workers=4)
        print(f"[bake] ok: {path}")
    print("[bake] MODELS_BAKED")


if __name__ == "__main__":
    main()
