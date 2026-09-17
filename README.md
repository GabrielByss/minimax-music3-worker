# minimax-music3-worker

Worker RunPod Serverless dla modelu [MiniMaxAI/MiniMax-Music3](https://huggingface.co/MiniMaxAI/MiniMax-Music3)
(pipeline diffusers >= 0.40.0). Silnik „Studio” w aplikacji **Music Ringtones for iPhone**; ten sam kontrakt
wejścia/wyjścia co worker ACE-Step (`acestep-ringtones-worker`), więc backend przełącza silniki polem `engine`.

Źródło prawdy: katalog `runpod/minimax` w repo `ringtones_repo` (to repo jest jego kopią do budowania obrazu).
Pełny opis (architektura, koszty, deploy, backend): `ringtones_repo/runpod/README.md`.

## Build

Actions → **Build MiniMax Music 3 worker image** → tag `v1`, `v2`, … → `ghcr.io/gabrielbyss/minimax-music3-worker:<tag>`.
Obraz z wypieczonymi wagami ma ~34 GB (7 warstw wag < 5 GB każda). Nowa wersja = nowy tag (RunPod cache'uje tagi na hostach).

## Kontrakt

`POST https://api.runpod.ai/v2/<ENDPOINT_ID>/run`

```json
{ "input": { "prompt": "Genre: upbeat pop. BPM: 120. ... Vocals: warm female lead ...",
             "lyrics": "[verse]\n...\n[chorus]\n...", "duration": 30, "vocal_language": "en",
             "inference_steps": 30, "audio_format": "mp3", "return_base64": true } }
```

Odpowiedź (`/status/{id}` → `COMPLETED`): `audio_base64` | `audio_url`, `format`, `sample_rate`, `duration`, `seed`,
`generation_seconds`, `model: "minimax-music3"`, `engine: "minimax"`, `attribution: "MiniMax-Music3"`.
`{"input":{"ping":true}}` → `{ok, gpu, vram_gb, offload}` (rozgrzewanie / health-check).

Licencja modelu (MiniMax-Music3 Community License): nazwa „MiniMax-Music3” musi być widoczna w UI produktu.
