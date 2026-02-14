import base64
import os
import re
from contextlib import asynccontextmanager
from typing import Any
from unicodedata import normalize

from dotenv import load_dotenv
import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROVIDER_CAPABILITIES: dict[str, dict[str, Any]] = {
    "openai": {
        "label": "OpenAI",
        "sizes": ["1024x1024", "1536x1024", "1024x1536"],
        "qualities": ["auto", "low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "requiresKey": "OPENAI_API_KEY",
    },
    "google": {
        "label": "Google",
        "sizes": ["1024x1024", "1280x720", "720x1280"],
        "qualities": ["standard", "hd"],
        "ratios": ["1:1", "16:9", "9:16"],
        "requiresKey": "GOOGLE_API_KEY",
    },
}


class GenerateResponse(BaseModel):
    provider: str
    size: str
    quality: str
    ratio: str
    image_data_url: str
    used_reference_images: int


class FilenameRequest(BaseModel):
    description: str


class FilenameResponse(BaseModel):
    filename: str


class TranscriptionResponse(BaseModel):
    text: str


class DescribeImageRequest(BaseModel):
    image_data_url: str
    source_text: str = ""
    language: str = ""


class DescribeImageResponse(BaseModel):
    description: str


class TtsRequest(BaseModel):
    text: str
    language: str = ""


def _ensure_api_key(env_name: str, provider_label: str) -> str:
    api_key = os.getenv(env_name)
    if not api_key:
        raise HTTPException(status_code=400, detail=f"{provider_label} API key not found in {env_name}")
    return api_key


def _to_data_url(image_b64: str, mime_type: str = "image/png") -> str:
    return f"data:{mime_type};base64,{image_b64}"


def _fallback_filename(description: str) -> str:
    base = normalize("NFKD", description).encode("ascii", "ignore").decode("ascii")
    base = re.sub(r"[^a-zA-Z0-9\s-]", "", base).strip().lower()
    base = re.sub(r"[-\s]+", "-", base)
    if not base:
        return "generated-image"
    return base[:80].strip("-") or "generated-image"


def _sanitize_filename(raw: str, fallback: str) -> str:
    cleaned = normalize("NFKD", raw).encode("ascii", "ignore").decode("ascii")
    cleaned = cleaned.strip().lower().replace(".png", "")
    cleaned = re.sub(r"[^a-zA-Z0-9\s-]", "", cleaned)
    cleaned = re.sub(r"[-\s]+", "-", cleaned).strip("-")
    return (cleaned or fallback)[:80]


def _safe_provider_error(provider_name: str, response: httpx.Response) -> HTTPException:
    try:
        payload = response.json()
    except ValueError:
        return HTTPException(
            status_code=502,
            detail=f"{provider_name} returned an unexpected error ({response.status_code}).",
        )

    error_obj = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error_obj, dict):
        message = error_obj.get("message", "")
        code = error_obj.get("code", "")
        if "moderation" in code or "safety" in message.lower():
            return HTTPException(
                status_code=422,
                detail=f"{provider_name}: Your prompt was blocked by the safety filter. Try rephrasing your description.",
            )
        if message:
            return HTTPException(
                status_code=502,
                detail=f"{provider_name}: {message}",
            )

    return HTTPException(
        status_code=502,
        detail=f"{provider_name} error ({response.status_code}). Please try again.",
    )


async def _read_reference_images(reference_images: list[UploadFile] | None) -> list[tuple[str, bytes, str]]:
    parsed: list[tuple[str, bytes, str]] = []
    for file in reference_images or []:
        content = await file.read()
        if not content:
            continue
        mime = file.content_type or "application/octet-stream"
        parsed.append((file.filename or "reference-image", content, mime))
    return parsed


async def _generate_with_openai(
    description: str,
    size: str,
    quality: str,
    reference_images: list[tuple[str, bytes, str]],
) -> str:
    api_key = _ensure_api_key("OPENAI_API_KEY", "OpenAI")
    headers = {"Authorization": f"Bearer {api_key}"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        if reference_images:
            data = {
                "model": "gpt-image-1",
                "prompt": description,
                "size": size,
                "quality": quality,
                "n": "1",
                "output_format": "png",
            }
            files = [
                ("image[]", (file_name, content, mime_type))
                for file_name, content, mime_type in reference_images
            ]
            response = await client.post(
                "https://api.openai.com/v1/images/edits",
                headers=headers,
                data=data,
                files=files,
            )
        else:
            response = await client.post(
                "https://api.openai.com/v1/images/generations",
                headers={**headers, "Content-Type": "application/json"},
                json={
                    "model": "gpt-image-1",
                    "prompt": description,
                    "size": size,
                    "quality": quality,
                    "output_format": "png",
                    "n": 1,
                },
            )

    if response.status_code >= 400:
        raise _safe_provider_error("OpenAI", response)

    payload = response.json()
    image_b64 = ((payload.get("data") or [{}])[0]).get("b64_json")
    if not image_b64:
        raise HTTPException(status_code=502, detail=f"OpenAI returned no image payload: {payload}")

    return _to_data_url(image_b64)


async def _generate_with_google(
    description: str,
    size: str,
    quality: str,
    ratio: str,
    reference_images: list[tuple[str, bytes, str]],
) -> str:
    api_key = _ensure_api_key("GOOGLE_API_KEY", "Google")

    parts: list[dict[str, Any]] = [
        {
            "text": (
                f"Generate one high quality image. Prompt: {description}. "
                f"Requested size: {size}. Requested quality: {quality}. Requested aspect ratio: {ratio}."
            )
        }
    ]

    for _, content, mime_type in reference_images:
        parts.append(
            {
                "inlineData": {
                    "mimeType": mime_type,
                    "data": base64.b64encode(content).decode("utf-8"),
                }
            }
        )

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-pro-image-preview:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{"parts": parts}],
                "generationConfig": {
                    "responseModalities": ["IMAGE", "TEXT"],
                },
            },
        )

    if response.status_code >= 400:
        raise _safe_provider_error("Google", response)

    payload = response.json()
    candidates = payload.get("candidates") or []
    for candidate in candidates:
        candidate_parts = ((candidate.get("content") or {}).get("parts")) or []
        for part in candidate_parts:
            inline_data = part.get("inline_data") or part.get("inlineData")
            if inline_data and inline_data.get("data"):
                mime = inline_data.get("mime_type") or inline_data.get("mimeType") or "image/png"
                return _to_data_url(inline_data["data"], mime)

    raise HTTPException(status_code=502, detail=f"Google returned no image payload: {payload}")


async def _suggest_filename_with_openai(description: str) -> str:
    fallback = _fallback_filename(description)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return fallback

    payload = {
        "model": "gpt-4.1-nano",
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return one short kebab-case filename only (no extension), max 8 words, "
                    "for the user's image request. No punctuation except hyphens."
                ),
            },
            {"role": "user", "content": description},
        ],
        "max_tokens": 30,
        "temperature": 0.2,
    }

    async with httpx.AsyncClient(timeout=12.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        return fallback

    body = response.json()
    content = (((body.get("choices") or [{}])[0]).get("message") or {}).get("content")
    if not content:
        return fallback

    return _sanitize_filename(content, fallback)


async def _describe_image_with_openai(image_data_url: str, source_text: str, language: str) -> str:
    api_key = _ensure_api_key("OPENAI_API_KEY", "OpenAI")
    language_instruction = (
        f"Use language '{language}'."
        if language
        else "Use the same language as SOURCE_TEXT."
    )
    payload = {
        "model": "gpt-4.1-nano",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You describe images very briefly. Return one sentence only, between 30 and 40 words. "
                    f"{language_instruction} Avoid markdown, lists, and preambles."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"SOURCE_TEXT: {source_text or 'N/A'}"},
                    {"type": "text", "text": "Describe this image now."},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            },
        ],
        "max_tokens": 120,
        "temperature": 0.3,
    }

    async with httpx.AsyncClient(timeout=18.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
        )

    if response.status_code >= 400:
        raise _safe_provider_error("OpenAI", response)

    body = response.json()
    description = ((((body.get("choices") or [{}])[0]).get("message") or {}).get("content") or "").strip()
    if not description:
        raise HTTPException(status_code=502, detail="OpenAI returned no image description.")

    return description


async def _synthesize_speech_with_openai(text: str, language: str) -> bytes:
    api_key = _ensure_api_key("OPENAI_API_KEY", "OpenAI")
    _ = language

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "tts-1-hd",
                "voice": "nova",
                "input": text,
                "response_format": "mp3",
            },
        )

    if response.status_code >= 400:
        raise _safe_provider_error("OpenAI", response)

    audio_bytes = response.content
    if not audio_bytes:
        raise HTTPException(status_code=502, detail="OpenAI returned no audio content.")

    return audio_bytes


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_dotenv()
    yield


app = FastAPI(title="BananaStore", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/providers")
async def get_providers() -> dict[str, Any]:
    providers = {}
    for provider_id, details in PROVIDER_CAPABILITIES.items():
        key_name = details["requiresKey"]
        providers[provider_id] = {
            **details,
            "hasKey": bool(os.getenv(key_name)),
        }
    return {"providers": providers}


@app.post("/api/suggest-filename", response_model=FilenameResponse)
async def suggest_filename(payload: FilenameRequest) -> FilenameResponse:
    description = payload.description.strip()
    if not description:
        return FilenameResponse(filename="generated-image")

    filename = await _suggest_filename_with_openai(description)
    return FilenameResponse(filename=filename)


@app.post("/api/transcribe-openai", response_model=TranscriptionResponse)
async def transcribe_openai(audio: UploadFile = File(...)) -> TranscriptionResponse:
    api_key = _ensure_api_key("OPENAI_API_KEY", "OpenAI")
    content = await audio.read()
    if not content:
        raise HTTPException(status_code=400, detail="No audio payload provided.")

    if len(content) > 8 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Audio file is too large. Keep it under 8MB.")

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {api_key}"},
            data={"model": "gpt-4o-mini-transcribe"},
            files={
                "file": (
                    audio.filename or "voice.webm",
                    content,
                    audio.content_type or "audio/webm",
                )
            },
        )

    if response.status_code >= 400:
        raise _safe_provider_error("OpenAI", response)

    payload = response.json()
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(status_code=502, detail="OpenAI returned no transcript text.")

    return TranscriptionResponse(text=text)


@app.post("/api/describe-image", response_model=DescribeImageResponse)
async def describe_image(payload: DescribeImageRequest) -> DescribeImageResponse:
    image_data_url = payload.image_data_url.strip()
    if not image_data_url.startswith("data:image/"):
        raise HTTPException(status_code=400, detail="image_data_url must be a valid data URL.")

    description = await _describe_image_with_openai(
        image_data_url=image_data_url,
        source_text=payload.source_text.strip(),
        language=payload.language.strip(),
    )
    return DescribeImageResponse(description=description)


@app.post("/api/tts-openai")
async def tts_openai(payload: TtsRequest) -> Response:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required.")

    audio_bytes = await _synthesize_speech_with_openai(text=text, language=payload.language.strip())
    return Response(content=audio_bytes, media_type="audio/mpeg")


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_image(
    provider: str = Form(...),
    description: str = Form(...),
    size: str = Form(...),
    quality: str = Form(...),
    ratio: str = Form(...),
    reference_images: list[UploadFile] | None = File(default=None),
) -> GenerateResponse:
    if provider not in PROVIDER_CAPABILITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    options = PROVIDER_CAPABILITIES[provider]
    if size not in options["sizes"]:
        raise HTTPException(status_code=400, detail=f"Unsupported size '{size}' for provider '{provider}'")
    if quality not in options["qualities"]:
        raise HTTPException(status_code=400, detail=f"Unsupported quality '{quality}' for provider '{provider}'")
    if ratio not in options["ratios"]:
        raise HTTPException(status_code=400, detail=f"Unsupported ratio '{ratio}' for provider '{provider}'")

    parsed_reference_images = await _read_reference_images(reference_images)
    reference_count = len(parsed_reference_images)

    if provider == "openai":
        image_data_url = await _generate_with_openai(
            description=description,
            size=size,
            quality=quality,
            reference_images=parsed_reference_images,
        )
    elif provider == "google":
        image_data_url = await _generate_with_google(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    return GenerateResponse(
        provider=provider,
        size=size,
        quality=quality,
        ratio=ratio,
        image_data_url=image_data_url,
        used_reference_images=reference_count,
    )


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
