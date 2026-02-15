# BananaStore

A sleek, self-hosted web app for AI image generation. Describe what you want, pick a provider, and get your image — all from a single page that runs locally.

![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-yellow)
![License: MIT](https://img.shields.io/badge/license-MIT-orange)

---

## Features

- **Multi-provider** — Generate images via OpenAI (gpt-image-1) or Google (Gemini), switchable in the UI
- **Reference images** — Upload photos or past generations as visual context for the AI
- **Voice input** — Describe your image by talking; speech is transcribed via OpenAI Whisper
- **AI narration** — Each generated image gets a spoken summary (OpenAI TTS) with an inline text card
- **Camera capture** — Snap a reference photo directly from your webcam or phone camera
- **Smart filenames** — Downloads get descriptive names suggested by an LLM
- **Lightbox viewer** — Zoom, pan, and browse through generated and reference images
- **Mobile-first tabs** — On small screens the UI switches to a tabbed layout (Create / Result) instead of scrolling
- **HTTPS proxy** — Optional nginx + self-signed cert for camera/mic access on LAN devices
- **Hot reload** — Uvicorn watches Python and static files, so changes appear instantly

## Quick start

```bash
# Clone
git clone git@github.com:Alyxion/BananaStore.git
cd BananaStore

# Install dependencies
poetry install

# Configure API keys
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY and/or GOOGLE_API_KEY

# Run
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8070 --reload
```

Open **http://localhost:8070** in your browser.

## HTTPS proxy (optional)

Camera and microphone access require a secure context on non-localhost origins. If you want to use BananaStore from another device on your network, start the included nginx reverse proxy:

```bash
# Generate a self-signed certificate
bash proxy/generate_cert.sh

# Start the proxy (exposes https://localhost:8453)
docker compose up -d
```

Then open `https://<your-ip>:8453` on your phone or tablet and accept the self-signed certificate.

## API keys

| Provider | Env variable | Used for |
|----------|-------------|----------|
| OpenAI | `OPENAI_API_KEY` | Image generation, voice transcription, image descriptions, TTS, filename suggestions |
| Google | `GOOGLE_API_KEY` | Image generation (Gemini) |

At least one key is required. The UI shows which providers have keys configured.

## Project structure

```
app/main.py          FastAPI backend — generation, transcription, TTS, describe endpoints
static/index.html    Single-page frontend
static/app.js        Client logic — form handling, voice, lightbox, mobile tabs
static/styles.css    Styles with responsive breakpoints
proxy/               Nginx HTTPS reverse proxy config + cert generator
docker-compose.yml   One-command proxy startup
```

## License

[MIT](LICENSE) — third-party licenses are listed in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
