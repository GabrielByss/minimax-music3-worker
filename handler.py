"""
MiniMax Music 3 — handler RunPod Serverless (kolejkowy: /run, /runsync, /status).

Ten sam kontrakt wejścia/wyjścia co runpod/serverless/handler.py (ACE-Step), żeby backend
(functions/v3/runpodAceStep.js) przełączał silniki polem `engine` bez zmiany finalizacji joba.
Model ładowany jest RAZ przy imporcie (cold start ~30–60 s przy wagach w obrazie), `handler(job)`
obsługuje kolejne zlecenia. Selftest: `MM3_MODE=selftest python handler.py` (używa test_input.json).

Wejście (job["input"]):
  prompt            str   opis muzyki; najlepiej "Structured Caption": Global Metadata (gatunek, BPM,
                          tonacja, emocje) / Vocal Details (płeć, barwa, styl) / Arrangement    [wymagane]
  lyrics            str   tekst; tagi [verse]/[chorus]/[bridge]/[instrumental]/[outro] KAŻDY w osobnej
                          linii; "" lub "[Instrumental]" => utwór instrumentalny
  duration          float sekundy, 5..MM3_MAX_DURATION (domyślnie 180). To GÓRNA granica –
                          LM może zakończyć utwór wcześniej tokenem końca                      domyślnie 30
  vocal_language    str   "en", "pl", ... – dopisywane do opisu jako język śpiewu           domyślnie "unknown"
  instrumental      bool  wymuś wersję bez wokalu                                          domyślnie false
  seed              int   -1 = losowy                                                      domyślnie -1
  inference_steps   int   kroki flow-matching na okno 8 s (5..60; 30 = referencyjne,
                          10–20 wyraźnie szybciej przy niewielkiej utracie jakości)       domyślnie MM3_DEFAULT_STEPS
  guidance_scale    float CFG etapu flow-matching (referencyjnie 1.7)                      domyślnie 1.7
  audio_format      str   mp3 | wav                                                        domyślnie mp3
  mp3_bitrate       str   np. "192k"                                                       domyślnie "192k"
  fade_out          float sekundy wyciszenia na końcu                                      domyślnie 0
  trim_to_duration  bool  przytnij wynik do `duration`                                     domyślnie true
  return_base64     bool  dołącz audio_base64 do outputu                                   domyślnie true (gdy brak upload_url)
  upload_url        str   presigned PUT URL (GCS/S3) – worker wgra tam plik
  upload_content_type str Content-Type dla PUT (domyślnie wg formatu)
  public_url        str   URL zwracany jako audio_url po udanym uploadzie
  ping              bool  szybki health-check bez generowania

Wyjście:
  audio_base64?, audio_url?, format, mime, sample_rate, duration, seed, size_bytes, model, engine,
  inference_steps, guidance_scale, generation_seconds, uploaded, upload_status?

Licencja modelu (MiniMax-Music3 Community License): produkt komercyjny musi widocznie pokazywać
nazwę "MiniMax-Music3" w interfejsie; powyżej 20 mln USD rocznego przychodu wymagana osobna zgoda.
"""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request

import numpy as np

MODE = os.environ.get("MM3_MODE", "serverless").strip().lower()
MODEL_ID = os.environ.get("MM3_MODEL_ID", "MiniMaxAI/MiniMax-Music3").strip()
OUTPUT_DIR = os.environ.get("MM3_OUTPUT_DIR", "/tmp/mm3-output")
ENGINE = "minimax"
MODEL_NAME = "minimax-music3"
ATTRIBUTION = "MiniMax-Music3"

MIME = {"mp3": "audio/mpeg", "wav": "audio/wav"}

# Język śpiewu (kody jak w ACE-Step / backendzie) -> nazwa do opisu
LANGUAGE_NAMES = {
    "en": "English", "pl": "Polish", "de": "German", "es": "Spanish", "fr": "French", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "uk": "Ukrainian", "cs": "Czech", "sk": "Slovak", "tr": "Turkish",
    "nl": "Dutch", "sv": "Swedish", "no": "Norwegian", "da": "Danish", "fi": "Finnish", "hu": "Hungarian",
    "ro": "Romanian", "bg": "Bulgarian", "hr": "Croatian", "el": "Greek", "he": "Hebrew", "ar": "Arabic",
    "hi": "Hindi", "bn": "Bengali", "id": "Indonesian", "vi": "Vietnamese", "th": "Thai", "tl": "Filipino",
    "ja": "Japanese", "ko": "Korean", "zh": "Mandarin Chinese", "fa": "Persian", "sw": "Swahili",
}


def log(msg: str) -> None:
    print(f"[minimax-worker] {msg}", flush=True)


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


DEFAULT_STEPS = int(env_float("MM3_DEFAULT_STEPS", 30))
DEFAULT_GUIDANCE = env_float("MM3_GUIDANCE_SCALE", 1.7)
MAX_DURATION = env_float("MM3_MAX_DURATION", 180.0)
MIN_DURATION = 5.0
HARD_MAX_DURATION = 360.0  # 9000 ramek @ 25 fps
OFFLOAD_MODE = os.environ.get("MM3_CPU_OFFLOAD", "auto").strip().lower()
OFFLOAD_BELOW_GB = env_float("MM3_OFFLOAD_BELOW_GB", 30.0)

# Wagi wypieczone w obrazie => nie dotykaj sieci przy starcie (cache HF w HF_HOME).
if env_bool("MM3_BAKED", True) and not os.environ.get("HF_HUB_OFFLINE"):
    os.environ["HF_HUB_OFFLINE"] = "1"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --------------------------------------------------------------------------------------
# Ładowanie modelu (raz, przy imporcie)
# --------------------------------------------------------------------------------------
BOOT_T0 = time.time()

import torch  # noqa: E402
from diffusers import ModularPipeline  # noqa: E402

try:  # nazwa modułu może się zmienić po zmergowaniu PR do diffusers
    from diffusers.guiders import ClassifierFreeGuidance  # noqa: E402
except ImportError:  # pragma: no cover
    from diffusers import ClassifierFreeGuidance  # type: ignore  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("MiniMax Music 3 wymaga CUDA (torch.cuda.is_available() == False)")

VRAM_GB = torch.cuda.get_device_properties(0).total_memory / 1e9
GPU_NAME = torch.cuda.get_device_name(0)
OFFLOAD = OFFLOAD_MODE in {"1", "true", "yes", "on"} or (OFFLOAD_MODE == "auto" and VRAM_GB < OFFLOAD_BELOW_GB)
log(f"boot: model={MODEL_ID} gpu={GPU_NAME} vram={VRAM_GB:.1f}GB offload={OFFLOAD} "
    f"offline={os.environ.get('HF_HUB_OFFLINE', '0')} hf_home={os.environ.get('HF_HOME', '')}")

# Wagi w obrazie: ładuj ze ścieżki lokalnej snapshotu HF i z local_files_only=True.
# Powód (diffusers 0.40): przy shardowanym checkpoincie z Hub (transformer/ ma 2 shardy) `_get_checkpoint_shard_files`
# woła `model_info()` po sieci, o ile nie dostanie `local_files_only=True`; pod HF_HUB_OFFLINE=1 kończy się to wyjątkiem,
# który `load_components` tylko loguje jako ostrzeżenie – pipeline startował z `transformer=None` i każde zlecenie padało
# po etapie AR z "'NoneType' object has no attribute 'dtype'" (incydent 2026-09-17, v1).
BAKED = env_bool("MM3_BAKED", True)
PIPE_SOURCE = MODEL_ID
LOAD_KWARGS: dict = {"dtype": torch.bfloat16}
if BAKED:
    LOAD_KWARGS["local_files_only"] = True
    # Katalog snapshotu w cache HF: przez ścieżkę zcache'owanego pliku indeksu (snapshot_download(local_files_only=True)
    # w huggingface_hub 1.x odrzuca snapshot jako "niekompletny", bo celowo nie mamy qwen_7B/ i *.pth).
    try:
        from huggingface_hub import hf_hub_download  # noqa: E402

        _index_path = hf_hub_download(MODEL_ID, "modular_model_index.json", local_files_only=True)
        PIPE_SOURCE = os.path.dirname(_index_path)  # .../snapshots/<commit>; pliki w środku to symlinki do blobs/
        if os.path.isdir(os.path.join(PIPE_SOURCE, "transformer")):
            LOAD_KWARGS["pretrained_model_name_or_path"] = PIPE_SOURCE
            log(f"weights: {PIPE_SOURCE}")
        else:  # pragma: no cover
            PIPE_SOURCE = MODEL_ID
            log(f"snapshot dir without transformer/ ({os.path.dirname(_index_path)}); loading by repo id with local_files_only=True")
    except Exception as exc:  # pragma: no cover
        PIPE_SOURCE = MODEL_ID
        log(f"snapshot lookup failed ({exc}); loading by repo id with local_files_only=True")

if OFFLOAD:
    # < ~30 GB VRAM (np. karty 24 GB): auto-offload komponentów na CPU (~22 GB VRAM, wolniej).
    from diffusers import ComponentsManager  # noqa: E402

    MANAGER = ComponentsManager()
    MANAGER.enable_auto_cpu_offload(device="cuda")
    PIPE = ModularPipeline.from_pretrained(PIPE_SOURCE, components_manager=MANAGER)
    PIPE.load_components(**LOAD_KWARGS)
else:
    PIPE = ModularPipeline.from_pretrained(PIPE_SOURCE)
    PIPE.load_components(**LOAD_KWARGS)
    PIPE.to("cuda")

# Twarda weryfikacja: brakujący komponent ma zatrzymać workera przy starcie, a nie po kilkudziesięciu sekundach GPU na zlecenie.
_specs = getattr(PIPE, "_component_specs", {}) or {}
_missing = [
    name for name, spec in _specs.items()
    if getattr(spec, "default_creation_method", "") == "from_pretrained" and getattr(PIPE, name, None) is None
]
if _missing:
    raise RuntimeError(f"MiniMax Music 3: nie załadowano komponentów {_missing} (źródło: {PIPE_SOURCE}, kwargs: {LOAD_KWARGS})")
_loaded = []
for name in _specs:
    comp = getattr(PIPE, name, None)
    if comp is None:
        continue
    dtype = getattr(comp, "dtype", None)
    device = getattr(comp, "device", None)
    _loaded.append(f"{name}={type(comp).__name__}" + (f"[{dtype}@{device}]" if dtype is not None else ""))
log("components: " + ", ".join(_loaded))

SAMPLE_RATE = int(PIPE.sampling_rate)
_current_guidance = 1.7  # wartość z konfiguracji pipeline'u (guider ClassifierFreeGuidance 1.7)
log(f"model loaded in {time.time() - BOOT_T0:.1f}s, sample_rate={SAMPLE_RATE}, frame_rate={PIPE.frame_rate}")


# --------------------------------------------------------------------------------------
# Pomocnicze
# --------------------------------------------------------------------------------------
def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _set_guidance(scale: float) -> None:
    global _current_guidance
    if abs(scale - _current_guidance) < 1e-6:
        return
    PIPE.update_components(guider=ClassifierFreeGuidance(guidance_scale=float(scale)))
    _current_guidance = float(scale)
    log(f"guidance_scale -> {scale}")


def _put_upload(url: str, data: bytes, content_type: str) -> int:
    req = urllib.request.Request(url, data=data, method="PUT")
    req.add_header("Content-Type", content_type)
    req.add_header("Content-Length", str(len(data)))
    with urllib.request.urlopen(req, timeout=120) as resp:  # nosec - URL podaje nasz backend
        return int(resp.status)


def _is_instrumental(lyrics: str) -> bool:
    s = lyrics.strip().lower()
    if not s:
        return True
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    # same tagi sekcji i wśród nich [instrumental] => bez tekstu do zaśpiewania
    return all(ln.startswith("[") and ln.endswith("]") for ln in lines) and any("instrumental" in ln for ln in lines)


def _instrumental_lyrics(duration: float) -> str:
    # wg testów społeczności każdy tag [instrumental] to ~9 s materiału
    tags = int(_clamp(round(duration / 9.0), 1, 40))
    return "\n".join(["[intro]"] + ["[instrumental]"] * tags + ["[outro]"])


def build_request(inp: dict) -> dict:
    prompt = str(inp.get("prompt") or inp.get("caption") or "").strip()[:2000]
    lyrics = str(inp.get("lyrics") or "").strip()[:4000]
    if not prompt:
        raise ValueError("Podaj 'prompt' (opis muzyki).")

    duration = float(inp.get("duration") or inp.get("audio_duration") or 30.0)
    duration = _clamp(duration, MIN_DURATION, min(MAX_DURATION, HARD_MAX_DURATION))

    instrumental = bool(inp.get("instrumental", False)) or _is_instrumental(lyrics)
    if instrumental:
        lyrics = _instrumental_lyrics(duration)
        prompt = "Instrumental track, no vocals, no singing, no lyrics.\n" + prompt

    language = str(inp.get("vocal_language") or inp.get("language") or "unknown").strip().lower()
    if not instrumental and language not in {"", "unknown"}:
        lang_name = LANGUAGE_NAMES.get(language, language)
        if lang_name.lower() not in prompt.lower():
            prompt = f"{prompt}\nVocals sung in {lang_name}, clear pronunciation."

    seed = int(inp.get("seed", -1))
    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "little")

    steps = int(_clamp(int(inp.get("inference_steps", DEFAULT_STEPS) or DEFAULT_STEPS), 5, 60))
    guidance = float(inp.get("guidance_scale", DEFAULT_GUIDANCE) or DEFAULT_GUIDANCE)

    audio_format = str(inp.get("audio_format") or "mp3").strip().lower()
    if audio_format not in MIME:
        audio_format = "mp3"

    return {
        "prompt": prompt,
        "lyrics": lyrics,
        "duration": duration,
        "instrumental": instrumental,
        "seed": seed,
        "steps": steps,
        "guidance": guidance,
        "format": audio_format,
        "mp3_bitrate": str(inp.get("mp3_bitrate") or "192k"),
        "fade_out": float(inp.get("fade_out", 0.0) or 0.0),
        "trim": bool(inp.get("trim_to_duration", True)),
    }


def run_pipeline(req: dict) -> np.ndarray:
    _set_guidance(req["guidance"])
    generator = torch.Generator("cuda").manual_seed(req["seed"])
    result = PIPE(
        prompt=req["prompt"],
        lyrics=req["lyrics"],
        audio_duration=float(req["duration"]),
        num_inference_steps=int(req["steps"]),
        generator=generator,
        output="audios",
    )
    audio = result[0] if isinstance(result, (list, tuple)) else result
    if isinstance(audio, torch.Tensor):
        audio = audio.float().cpu().numpy()
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 3:  # [batch, channels, samples]
        audio = audio[0]
    if audio.ndim == 1:
        audio = audio[:, None]
    elif audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:  # [channels, samples] -> [samples, channels]
        audio = audio.T
    return np.clip(audio, -1.0, 1.0)


def postprocess(audio: np.ndarray, req: dict) -> np.ndarray:
    if req["trim"]:
        max_samples = int(req["duration"] * SAMPLE_RATE)
        if audio.shape[0] > max_samples:
            audio = audio[:max_samples]
    fade = req["fade_out"]
    if fade > 0 and audio.shape[0] > 1:
        n = int(min(fade * SAMPLE_RATE, audio.shape[0]))
        if n > 1:
            ramp = np.linspace(1.0, 0.0, n, dtype=np.float32)[:, None]
            audio = audio.copy()
            audio[-n:] *= ramp
    return audio


def encode(audio: np.ndarray, req: dict, work_dir: str) -> tuple[bytes, str]:
    import soundfile as sf

    wav_path = os.path.join(work_dir, "out.wav")
    sf.write(wav_path, audio, SAMPLE_RATE, subtype="PCM_16")
    if req["format"] == "wav":
        with open(wav_path, "rb") as fh:
            return fh.read(), "wav"

    mp3_path = os.path.join(work_dir, "out.mp3")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", wav_path, "-codec:a", "libmp3lame",
             "-b:a", req["mp3_bitrate"], mp3_path],
            check=True, timeout=300,
        )
    else:  # libsndfile >= 1.1 potrafi zapisać MP3
        sf.write(mp3_path, audio, SAMPLE_RATE, format="MP3")
    with open(mp3_path, "rb") as fh:
        return fh.read(), "mp3"


def generate(inp: dict, job_id: str) -> dict:
    req = build_request(inp)
    work_dir = os.path.join(OUTPUT_DIR, job_id)
    os.makedirs(work_dir, exist_ok=True)
    t0 = time.time()
    try:
        log(f"job {job_id}: duration={req['duration']}s steps={req['steps']} seed={req['seed']} "
            f"instrumental={req['instrumental']} lyrics_chars={len(req['lyrics'])}")
        audio = run_pipeline(req)
        audio = postprocess(audio, req)
        elapsed = time.time() - t0
        data, fmt = encode(audio, req, work_dir)
        actual_duration = audio.shape[0] / SAMPLE_RATE
        content_type = str(inp.get("upload_content_type") or MIME[fmt])

        out = {
            "format": fmt,
            "mime": content_type,
            "sample_rate": SAMPLE_RATE,
            "duration": round(actual_duration, 3),
            "seed": req["seed"],
            "size_bytes": len(data),
            "model": MODEL_NAME,
            "engine": ENGINE,
            "attribution": ATTRIBUTION,
            "instrumental": req["instrumental"],
            "inference_steps": req["steps"],
            "guidance_scale": req["guidance"],
            "generation_seconds": round(elapsed, 3),
            "uploaded": False,
        }

        upload_url = inp.get("upload_url")
        if upload_url:
            status = _put_upload(str(upload_url), data, content_type)
            out["uploaded"] = 200 <= status < 300
            out["upload_status"] = status
            if out["uploaded"] and inp.get("public_url"):
                out["audio_url"] = str(inp["public_url"])

        want_b64 = inp.get("return_base64")
        if want_b64 is None:
            want_b64 = not bool(upload_url)
        if want_b64:
            out["audio_base64"] = base64.b64encode(data).decode("ascii")
        log(f"job {job_id}: done in {elapsed:.1f}s, audio {actual_duration:.1f}s, {len(data)} bytes")
        return out
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        torch.cuda.empty_cache()


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    if inp.get("ping"):
        return {
            "ok": True, "model": MODEL_NAME, "engine": ENGINE, "gpu": GPU_NAME, "vram_gb": round(VRAM_GB, 1),
            "offload": OFFLOAD, "uptime_s": round(time.time() - BOOT_T0, 1),
        }
    try:
        return generate(inp, str(job.get("id") or f"local-{int(time.time())}"))
    except ValueError as exc:
        return {"error": str(exc)}
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover
        torch.cuda.empty_cache()
        return {"error": f"CUDA OOM ({VRAM_GB:.0f} GB VRAM, offload={OFFLOAD}): {exc}"}
    except Exception as exc:  # pragma: no cover
        log(traceback.format_exc())
        return {"error": f"{type(exc).__name__}: {exc}"}


if __name__ == "__main__":
    if MODE in {"selftest", "pod"}:
        test_path = os.environ.get("MM3_TEST_INPUT", os.path.join(os.path.dirname(__file__), "test_input.json"))
        with open(test_path, "r", encoding="utf-8") as fh:
            job = json.load(fh)
        job.setdefault("id", "selftest")
        res = handler(job)
        b64 = res.pop("audio_base64", None)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        if b64:
            out_file = os.path.join(os.getcwd(), f"selftest.{res.get('format', 'mp3')}")
            with open(out_file, "wb") as fh:
                fh.write(base64.b64decode(b64))
            print(f"zapisano {out_file}")
    else:
        import runpod  # noqa: E402

        runpod.serverless.start({"handler": handler})
