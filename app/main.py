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
        "qualities": ["auto", "low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Photo", "Vector"],
        "formatQualities": {"Photo": ["auto", "low", "medium", "high"], "Vector": ["low", "medium", "high"]},
        "requiresKey": "OPENAI_API_KEY",
    },
    "google": {
        "label": "Google",
        "qualities": ["standard", "hd"],
        "ratios": ["1:1", "16:9", "9:16"],
        "ratioSizes": {"1:1": "1024x1024", "16:9": "1280x720", "9:16": "720x1280"},
        "formats": ["Photo"],
        "requiresKey": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "label": "Anthropic",
        "qualities": ["low", "medium", "high"],
        "ratios": ["1:1", "3:2", "2:3"],
        "ratioSizes": {"1:1": "1024x1024", "3:2": "1536x1024", "2:3": "1024x1536"},
        "formats": ["Vector"],
        "requiresKey": "ANTHROPIC_API_KEY",
    },
}

SVG_QUALITY_HINTS: dict[str, str] = {
    "low": (
        "Style: clean flat design. Use simple geometric shapes, solid fills, minimal paths. "
        "No gradients or filters. Think app-icon or logo level simplicity."
    ),
    "medium": (
        "Style: polished vector illustration. Use layered shapes with gradients for depth, "
        "highlights, and shadows. Build complex objects by composing many precise smaller shapes "
        "(e.g. a shoe = sole shape + upper shape + lace loops + tongue + stitching lines). "
        "Each distinct part of the subject should be its own shape with correct proportions."
    ),
    "high": (
        "Style: detailed, near-realistic vector illustration. Construct the subject from many "
        "precise, anatomically/structurally correct sub-shapes. Every distinct part must be its "
        "own carefully shaped path (e.g. a running shoe needs: outsole, midsole, upper panel, "
        "toe box, heel counter, tongue, lace eyelets, individual laces, swoosh/logo area, "
        "pull tab, stitching lines — each as separate paths with correct proportions). "
        "Use linear and radial gradients for realistic shading and material texture. "
        "Add subtle highlights, shadow layers, and edge detail. "
        "Stay under ~1000 path elements for performance."
    ),
}

SVG_QUALITY_TOKENS: dict[str, int] = {
    "low": 4000,
    "medium": 8000,
    "high": 16000,
}


class GenerateResponse(BaseModel):
    provider: str
    size: str
    quality: str
    ratio: str
    format: str
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
    cleaned = cleaned.strip().lower().replace(".png", "").replace(".svg", "")
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


async def _read_reference_images(
    reference_images: list[UploadFile] | None,
) -> tuple[list[tuple[str, bytes, str]], list[str]]:
    parsed: list[tuple[str, bytes, str]] = []
    svg_sources: list[str] = []
    for file in reference_images or []:
        content = await file.read()
        if not content:
            continue
        mime = file.content_type or "application/octet-stream"
        name = file.filename or "reference-image"
        if mime == "image/svg+xml" or name.endswith(".svg"):
            try:
                svg_sources.append(content.decode("utf-8", errors="ignore"))
            except Exception:
                pass
            continue
        parsed.append((name, content, mime))
    return parsed, svg_sources


SVG_SYSTEM_PROMPT = (
    "You are an expert SVG illustrator who creates structurally accurate vector art. "
    "Output ONLY raw SVG markup — no markdown fences, no explanation, no extra text.\n\n"
    "Technical requirements:\n"
    '- xmlns="http://www.w3.org/2000/svg" attribute, viewBox "0 0 {width} {height}"\n'
    "- No external resources (images, fonts, stylesheets) — inline styles only\n"
    "- Web-safe fonts only (Arial, Helvetica, Georgia, Verdana, sans-serif, serif)\n\n"
    "Critical illustration rules:\n"
    "- BEFORE writing SVG, mentally decompose the subject into its real structural parts. "
    "A shoe is not a dome — it has a flat sole, a low-profile upper, laces, a tongue, etc. "
    "A car is not a blob — it has wheels, windows, a hood, doors, etc.\n"
    "- Get the PROPORTIONS and SILHOUETTE right first. The overall outline must be "
    "recognizable as the subject even without color or detail.\n"
    "- Build from back to front using layered shapes — background elements first, "
    "foreground details on top.\n"
    "- Use <path> with precise d attributes for organic curves. Use basic shapes "
    "(rect, circle, ellipse) where geometrically appropriate.\n\n"
    "{quality_hint}\n\n"
    "If reference images are provided, study their structure and proportions carefully."
)


def _parse_svg_dimensions(size: str) -> tuple[int, int]:
    parts = size.split("x")
    return int(parts[0]), int(parts[1])


def _extract_svg(text: str) -> str:
    match = re.search(r"<svg[\s\S]*?</svg>", text, re.IGNORECASE)
    if not match:
        raise HTTPException(status_code=502, detail="Model did not return valid SVG markup.")
    svg = match.group(0)
    if "xmlns" not in svg:
        svg = svg.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
    return svg


async def _generate_svg_with_openai(
    description: str,
    size: str,
    quality: str,
    ratio: str,
    reference_images: list[tuple[str, bytes, str]],
    svg_sources: list[str] | None = None,
) -> str:
    api_key = _ensure_api_key("OPENAI_API_KEY", "OpenAI")
    width, height = _parse_svg_dimensions(size)
    quality_hint = SVG_QUALITY_HINTS.get(quality, SVG_QUALITY_HINTS["medium"])
    max_tokens = SVG_QUALITY_TOKENS.get(quality, 8000)
    system_prompt = SVG_SYSTEM_PROMPT.format(width=width, height=height, quality_hint=quality_hint)

    user_content: list[dict[str, Any]] = []
    for _, content, mime_type in reference_images:
        b64 = base64.b64encode(content).decode("utf-8")
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
        })
    prompt_text = f"Create an SVG illustration: {description}. Target size: {size}, aspect ratio: {ratio}."
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt_text += f"\n\nReference SVG {i} source (adjust as needed):\n{src}"
    user_content.append({"type": "text", "text": prompt_text})

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-5.2",
                "max_completion_tokens": max_tokens,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        )

    if response.status_code >= 400:
        raise _safe_provider_error("OpenAI", response)

    payload = response.json()
    text = (((payload.get("choices") or [{}])[0]).get("message") or {}).get("content") or ""
    svg = _extract_svg(text)
    svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{svg_b64}"


async def _generate_with_anthropic(
    description: str,
    size: str,
    quality: str,
    ratio: str,
    reference_images: list[tuple[str, bytes, str]],
    svg_sources: list[str] | None = None,
) -> str:
    api_key = _ensure_api_key("ANTHROPIC_API_KEY", "Anthropic")
    width, height = _parse_svg_dimensions(size)
    quality_hint = SVG_QUALITY_HINTS.get(quality, SVG_QUALITY_HINTS["medium"])
    max_tokens = SVG_QUALITY_TOKENS.get(quality, 8000)
    system_prompt = SVG_SYSTEM_PROMPT.format(width=width, height=height, quality_hint=quality_hint)

    ANTHROPIC_IMAGE_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
    user_content: list[dict[str, Any]] = []
    for _, content, mime_type in reference_images:
        if mime_type not in ANTHROPIC_IMAGE_MIMES:
            continue
        b64 = base64.b64encode(content).decode("utf-8")
        user_content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mime_type, "data": b64},
        })
    prompt_text = f"Create an SVG illustration: {description}. Target size: {size}, aspect ratio: {ratio}."
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt_text += f"\n\nReference SVG {i} source (adjust as needed):\n{src}"
    user_content.append({"type": "text", "text": prompt_text})

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
            json={
                "model": "claude-opus-4-6",
                "max_tokens": max_tokens,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_content},
                ],
            },
        )

    if response.status_code >= 400:
        raise _safe_provider_error("Anthropic", response)

    payload = response.json()
    text_parts = [
        block.get("text", "")
        for block in (payload.get("content") or [])
        if block.get("type") == "text"
    ]
    full_text = "\n".join(text_parts)
    svg = _extract_svg(full_text)
    svg_b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    return f"data:image/svg+xml;base64,{svg_b64}"


async def _generate_with_openai(
    description: str,
    size: str,
    quality: str,
    reference_images: list[tuple[str, bytes, str]],
    svg_sources: list[str] | None = None,
) -> str:
    api_key = _ensure_api_key("OPENAI_API_KEY", "OpenAI")
    headers = {"Authorization": f"Bearer {api_key}"}
    prompt = description
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt += f"\n\nReference SVG {i} source (use as visual inspiration):\n{src}"

    async with httpx.AsyncClient(timeout=120.0) as client:
        if reference_images:
            data = {
                "model": "gpt-image-1",
                "prompt": prompt,
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
                    "prompt": prompt,
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
    svg_sources: list[str] | None = None,
) -> str:
    api_key = _ensure_api_key("GOOGLE_API_KEY", "Google")

    prompt_text = (
        f"Generate one high quality image. Prompt: {description}. "
        f"Requested size: {size}. Requested quality: {quality}. Requested aspect ratio: {ratio}."
    )
    if svg_sources:
        for i, src in enumerate(svg_sources, 1):
            prompt_text += f"\n\nReference SVG {i} source (use as visual inspiration):\n{src}"

    parts: list[dict[str, Any]] = [{"text": prompt_text}]

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
                    "You are a friendly artist commenting on an image you just created for someone. "
                    "Speak naturally in first person — mention what you did with their idea, "
                    "highlight a detail you're proud of, or note a creative choice you made. "
                    "Keep it to one or two sentences, 25–40 words. Be warm but not over the top. "
                    f"{language_instruction} Avoid markdown, lists, and preambles."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"SOURCE_TEXT: {source_text or 'N/A'}"},
                    {"type": "text", "text": "Comment on this image you just created for the user."},
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
    quality: str = Form(...),
    ratio: str = Form(...),
    format: str = Form("Photo"),
    reference_images: list[UploadFile] | None = File(default=None),
) -> GenerateResponse:
    if provider not in PROVIDER_CAPABILITIES:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    options = PROVIDER_CAPABILITIES[provider]
    format_qualities = (options.get("formatQualities") or {}).get(format)
    allowed_qualities = format_qualities or options["qualities"]
    if quality not in allowed_qualities:
        raise HTTPException(status_code=400, detail=f"Unsupported quality '{quality}' for provider '{provider}'")
    if ratio not in options["ratios"]:
        raise HTTPException(status_code=400, detail=f"Unsupported ratio '{ratio}' for provider '{provider}'")
    if format not in options["formats"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{format}' for provider '{provider}'")

    size = options["ratioSizes"].get(ratio, "1024x1024")
    parsed_reference_images, svg_sources = await _read_reference_images(reference_images)
    reference_count = len(parsed_reference_images) + len(svg_sources)

    if format == "Vector" and provider == "openai":
        image_data_url = await _generate_svg_with_openai(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    elif format == "Vector" and provider == "anthropic":
        image_data_url = await _generate_with_anthropic(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    elif provider == "openai":
        image_data_url = await _generate_with_openai(
            description=description,
            size=size,
            quality=quality,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    elif provider == "google":
        image_data_url = await _generate_with_google(
            description=description,
            size=size,
            quality=quality,
            ratio=ratio,
            reference_images=parsed_reference_images,
            svg_sources=svg_sources,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported provider: {provider}")

    return GenerateResponse(
        provider=provider,
        size=size,
        quality=quality,
        ratio=ratio,
        format=format,
        image_data_url=image_data_url,
        used_reference_images=reference_count,
    )


@app.get("/")
async def root() -> FileResponse:
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")
