# =============================================================================
# MiniMax Music 3 — worker RunPod Serverless (queue-based, handler.py)
#
# Baza: oficjalny obraz PyTorch (CUDA 12.8, Python 3.11). Dokładamy diffusers >= 0.40.0 (pierwsze
# wydanie z pipeline'em MiniMax-Music3, PR huggingface/diffusers#14456 zmergowany 2026-08-13),
# transformers 5.x, SDK runpod, ffmpeg.
# Wagi (tylko komponenty pipeline'u diffusers, ~28,5 GB z 57 GB repo HF) domyślnie WYPIEKANE
# do obrazu w 7 warstwach (< 5 GB każda; limit warstwy w GHCR to 10 GB, mniejsze warstwy RunPod
# pobiera równolegle), żeby cold start nie pobierał ich z HuggingFace. Trafiają do cache HF (/app/hf),
# a handler ładuje je offline (MM3_BAKED=true => HF_HUB_OFFLINE=1).
#
# Build (x86_64!):
#   docker build --platform linux/amd64 -t <registry>/minimax-music3-worker:v1 runpod/minimax
#   docker build --platform linux/amd64 --build-arg BAKE_MODELS=false ...   # wagi z network volume
# =============================================================================
ARG BASE_IMAGE=pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

ARG BAKE_MODELS=true
ARG HF_TOKEN=""
# Wydanie z PyPI. Model card wskazuje commit z PR (dafe3733…), ale jego zmiany (w tym workaround offloadu
# w etapie AR) są w v0.40.0. Alternatywa: DIFFUSERS_SPEC="git+https://github.com/huggingface/diffusers@<sha>"
ARG DIFFUSERS_SPEC="diffusers==0.40.0"

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HOME=/app/hf

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt \
    && pip install "${DIFFUSERS_SPEC}" \
    && python -c "import diffusers, transformers, huggingface_hub; from diffusers.modular_pipelines import minimax_music3; print('diffusers', diffusers.__version__, 'transformers', transformers.__version__, 'hf_hub', huggingface_hub.__version__)"

# Wypiekanie wag w częściach (osobne warstwy < 5 GB: lm1..lm4 = Qwen3-8B, fm1..fm2 = flow matching, rest = dekodery/tokenizer)
COPY download_models.py /app/download_models.py
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py lm1; else echo "BAKE_MODELS=false"; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py lm2; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py lm3; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py lm4; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py fm1; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py fm2; fi
RUN if [ "$BAKE_MODELS" = "true" ]; then HF_TOKEN="$HF_TOKEN" python /app/download_models.py rest; fi

COPY handler.py /app/handler.py

# MM3_BAKED=true => handler ustawia HF_HUB_OFFLINE=1 (wagi z obrazu). Dla network volume ustaw w template:
#   MM3_BAKED=false, HF_HOME=/runpod-volume/hf  (pierwszy worker pobierze ~28 GB na wolumen)
ENV MM3_MODE=serverless \
    MM3_BAKED=${BAKE_MODELS} \
    MM3_MODEL_ID=MiniMaxAI/MiniMax-Music3 \
    MM3_CPU_OFFLOAD=auto \
    MM3_DEFAULT_STEPS=30 \
    MM3_MAX_DURATION=180 \
    MM3_OUTPUT_DIR=/tmp/mm3-output

CMD ["python", "-u", "/app/handler.py"]
