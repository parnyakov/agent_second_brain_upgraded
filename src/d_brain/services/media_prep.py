"""Prepare a saved attachment so the brain can read it cheaply and safely.

Background: Telegram hands `document`
attachments over UNCOMPRESSED — a phone camera original is ~12x bigger and
3-4x higher resolution than the same picture sent as a `[photo]` (Telegram
downscales those to ~1280px itself). Handing such a file to the brain "as
is" made a single turn run long enough to wedge the delivery channel twice
(2026-08-30 and 2026-08-31), which then escalated into a `delivery_guard`
"channel broken" alert.

This module is the one place that decides WHAT the brain should actually
open for a given attachment:

* heavy images  → a downscaled derivative (<=1568px long side);
* big PDFs      → a pypdf-extracted text sidecar, with a page-limited
                  native-Read fallback for scans;
* video/audio/binary → an honest instruction that says the file cannot be
                  processed and must NOT be read.

Two hard rules:

1. The original in ``attachments/`` is NEVER mutated. This follows the
   owner's rule that AI photo-session workflows require untouched
   originals.
2. Any failure degrades to today's behavior (original path + the generic
   instruction) — never to an exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

# --- Constants (fixed by the plan, not tuned at call sites) ---------------

#: Ceiling of useful vision resolution — a longer side buys nothing.
MODEL_LONG_SIDE = 1568
#: Above this an image is downscaled even if it already fits MODEL_LONG_SIDE.
MAX_IMAGE_BYTES = 1_500_000
#: JPEG quality for derivatives.
JPEG_QUALITY = 85
#: A derivative is only worth keeping if it is meaningfully smaller than the
#: original — see ``_prepare_image``.
DERIVATIVE_MAX_RATIO = 0.9

#: A PDF this small is read natively, no text extraction.
PDF_DIRECT_MAX_PAGES = 5
PDF_DIRECT_MAX_BYTES = 2_000_000
#: Less extracted text than this across the whole document ⇒ it's a scan.
PDF_SCAN_TEXT_CHARS = 200

#: Text documents bigger than this must be read in chunks, not whole.
TEXT_CHUNK_BYTES = 256 * 1024
TEXT_READ_LIMIT_LINES = 500

_BASE_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}
_HEIF_EXTS = {"heic", "heif"}

try:  # pillow-heif is declared in pyproject; stay standing if it isn't there
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIF_SUPPORTED = True
except Exception:  # noqa: BLE001 — optional decoder, never fatal
    HEIF_SUPPORTED = False

#: Extensions we treat as images (HEIC/HEIF only when the decoder loaded).
IMAGE_EXTS = _BASE_IMAGE_EXTS | (_HEIF_EXTS if HEIF_SUPPORTED else set())

PDF_EXTS = {"pdf"}
VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "webm", "m4v", "3gp", "gif"}
AUDIO_EXTS = {"mp3", "wav", "m4a", "ogg", "oga", "flac", "aac", "opus", "wma"}
TEXT_EXTS = {"txt", "md", "csv", "json", "log", "yaml", "yml", "xml", "html"}

VIDEO_KINDS = {"video", "animation", "video_note"}
AUDIO_KINDS = {"audio"}

_SAVE_FORMATS = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "bmp": "BMP",
    "tif": "TIFF",
    "tiff": "TIFF",
}

_TAIL = "сохрани суть в память по правилам vault и кратко ответь, что сохранил."
_TAIL_CAP = _TAIL[0].upper() + _TAIL[1:]

#: Exactly today's wording — the safe fallback for anything unrecognised or
#: for any failure inside this module.
DEFAULT_INSTRUCTION = (
    "Прочитай файл (Read поддерживает изображения и PDF; для видео/аудио "
    f"опиши по подписи и контексту), {_TAIL}"
)

_ONLY_ORIGINAL = (
    "В память и заметки ссылайся ТОЛЬКО на оригинал — производная копия "
    "временная и в vault не индексируется."
)


@dataclass
class MediaPrep:
    """What the brain should open, and how it is told to open it.

    ``model_rel_path`` is a vault-relative path to a derivative (a downscaled
    image or an extracted-text sidecar), or ``None`` to use the original.
    ``instruction`` is the ready-made instruction line for the prompt.
    ``meta`` is a short human note for the prompt header ("36 страниц").
    """

    model_rel_path: str | None = None
    instruction: str = DEFAULT_INSTRUCTION
    meta: str = ""


# --- Public entry point ---------------------------------------------------


def prepare_for_model(
    vault_path: Path | str, rel_path: str, kind: str, ext: str
) -> MediaPrep:
    """Decide what the brain reads for the attachment at ``rel_path``.

    Never raises: any failure logs a warning and degrades to today's
    behavior (original path + the generic instruction).
    """
    try:
        return _prepare(
            Path(vault_path), rel_path, kind, (ext or "").lower().lstrip(".")
        )
    except Exception:  # noqa: BLE001 — degradation must never break a reply
        logger.warning(
            "media_prep failed for %s, using original", rel_path, exc_info=True
        )
        return MediaPrep()


def _prepare(vault_path: Path, rel_path: str, kind: str, ext: str) -> MediaPrep:
    if kind in VIDEO_KINDS or ext in VIDEO_EXTS:
        return MediaPrep(instruction=video_instruction(rel_path))
    if kind in AUDIO_KINDS or ext in AUDIO_EXTS:
        return MediaPrep(instruction=audio_instruction(rel_path))
    if ext in IMAGE_EXTS:
        return _prepare_image(vault_path, rel_path, ext)
    if ext in PDF_EXTS:
        return _prepare_pdf(vault_path, rel_path)
    if ext in TEXT_EXTS:
        return _prepare_text(vault_path, rel_path)
    return MediaPrep(instruction=binary_instruction(rel_path))


# --- Instructions ---------------------------------------------------------


def video_instruction(rel_path: str) -> str:
    return (
        f"Видео обработать не могу. Файл сохранён: {rel_path}. "
        "НЕ читай этот файл (это бинарник, чтение дорогое и бесполезное). "
        "Зафиксируй факт, имя файла и подпись в память по правилам vault "
        "и кратко ответь."
    )


def audio_instruction(rel_path: str) -> str:
    return (
        f"Audio-файлы не транскрибирую (голосовые сообщения идут отдельным "
        f"путём). Файл сохранён: {rel_path}. НЕ читай этот файл. "
        "Зафиксируй факт, имя файла и подпись в память по правилам vault "
        "и кратко ответь."
    )


def binary_instruction(rel_path: str) -> str:
    return (
        f"Файл сохранён: {rel_path}. НЕ читай бинарный файл — содержимое "
        "нечитаемо, попытка дорогая. Зафиксируй факт, имя файла и подпись "
        "в память по правилам vault и кратко ответь."
    )


def _image_plain_instruction() -> str:
    return f"Прочитай изображение (Read поддерживает картинки), {_TAIL}"


def _image_downscaled_instruction(rel_path: str, model_rel_path: str) -> str:
    return (
        f"Читай уменьшенную копию: {model_rel_path}. Оригинал {rel_path} "
        "НЕ открывай — он тяжёлый и способен подвесить сессию. "
        f"{_ONLY_ORIGINAL} {_TAIL_CAP}"
    )


def _pdf_direct_instruction() -> str:
    return f"PDF небольшой — прочитай его напрямую (Read поддерживает PDF), {_TAIL}"


def _pdf_text_instruction(rel_path: str, sidecar_rel: str) -> str:
    return (
        f"Читай извлечённый текст: {sidecar_rel}. Оригинал {rel_path} целиком "
        "не открывай. Если критична вёрстка или картинки — Read оригинала, "
        f"не более {PDF_DIRECT_MAX_PAGES} страниц за раз. {_ONLY_ORIGINAL} "
        f"{_TAIL_CAP}"
    )


def _pdf_scan_instruction(rel_path: str) -> str:
    return (
        f"В PDF нет текстового слоя — это скан. Read оригинала {rel_path}, "
        f"страницы 1-{PDF_DIRECT_MAX_PAGES}, дальше только малыми порциями "
        f"по необходимости. {_TAIL_CAP}"
    )


def _text_chunked_instruction(rel_path: str) -> str:
    return (
        f"Текстовый файл большой — НЕ читай целиком. Read {rel_path} порциями "
        f"по ~{TEXT_READ_LIMIT_LINES} строк (limit/offset), пока не поймёшь "
        f"суть. {_TAIL_CAP}"
    )


# --- Helpers --------------------------------------------------------------


def derived_rel_path(rel_path: str, suffix: str) -> str:
    """``attachments/D/img-1.jpg`` + ``-model.jpg`` →
    ``attachments/D/derived/img-1-model.jpg``."""
    p = PurePosixPath(rel_path)
    return str(p.parent / "derived" / f"{p.stem}{suffix}")


def _derived_target(vault_path: Path, rel_path: str, suffix: str) -> tuple[Path, str]:
    rel = derived_rel_path(rel_path, suffix)
    target = vault_path / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, rel


def _human(size: int) -> str:
    if size >= 1_000_000:
        return f"{size / 1_000_000:.1f} МБ"
    return f"{max(1, round(size / 1000))} КБ"


# --- Images ---------------------------------------------------------------


def _prepare_image(vault_path: Path, rel_path: str, ext: str) -> MediaPrep:
    from PIL import Image, ImageOps

    src = vault_path / rel_path
    size = src.stat().st_size

    def _use_original(width: int, height: int) -> MediaPrep:
        """No derivative — the brain reads the original, as it always has."""
        return MediaPrep(
            instruction=_image_plain_instruction(),
            meta=f"{width}×{height}, {_human(size)}",
        )

    with Image.open(src) as opened:
        image = ImageOps.exif_transpose(opened) or opened
        width, height = image.size
        oversized = max(width, height) > MODEL_LONG_SIDE or size > MAX_IMAGE_BYTES
        if not oversized:
            # A normal Telegram [photo] lands here and is passed through
            # untouched — no derivative, behavior unchanged.
            return _use_original(width, height)

        out_ext = "jpg" if ext in _HEIF_EXTS else ext
        fmt = _SAVE_FORMATS.get(out_ext, "JPEG")
        image.thumbnail((MODEL_LONG_SIDE, MODEL_LONG_SIDE), Image.Resampling.LANCZOS)

        save_kwargs: dict[str, object] = {}
        if fmt == "JPEG":
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            save_kwargs = {"quality": JPEG_QUALITY, "optimize": True}
        elif fmt == "PNG":
            save_kwargs = {"optimize": True}

        target, rel = _derived_target(vault_path, rel_path, f"-model.{out_ext}")
        image.save(target, format=fmt, **save_kwargs)
        new_w, new_h = image.size
        derived_size = target.stat().st_size

    # ``thumbnail()`` only ever shrinks by DIMENSION. An image that tripped
    # the byte threshold ALONE (e.g. a 900×900 incompressible PNG) is left at
    # its original size, and the re-encode can come out no smaller — or
    # bigger. Keeping that would make the prompt lie ("read this smaller
    # copy, don't open the heavy original") and waste disk, so drop it and
    # fall back to the plain use-the-original behavior.
    if derived_size >= size * DERIVATIVE_MAX_RATIO:
        logger.info(
            "media_prep: derivative for %s not meaningfully smaller "
            "(%d vs %d bytes) — using the original",
            rel_path,
            derived_size,
            size,
        )
        target.unlink(missing_ok=True)
        return _use_original(width, height)

    return MediaPrep(
        model_rel_path=rel,
        instruction=_image_downscaled_instruction(rel_path, rel),
        meta=(
            f"{width}×{height} {_human(size)} → {new_w}×{new_h} "
            f"{_human(derived_size)}"
        ),
    )


# --- PDF ------------------------------------------------------------------


def _prepare_pdf(vault_path: Path, rel_path: str) -> MediaPrep:
    from pypdf import PdfReader

    src = vault_path / rel_path
    size = src.stat().st_size
    reader = PdfReader(src)
    if reader.is_encrypted:
        # An empty-password PDF still decrypts; a real one raises and the
        # caller's fallback takes over.
        reader.decrypt("")
    pages = len(reader.pages)
    meta = f"{pages} стр., {_human(size)}"

    if pages <= PDF_DIRECT_MAX_PAGES and size <= PDF_DIRECT_MAX_BYTES:
        return MediaPrep(instruction=_pdf_direct_instruction(), meta=meta)

    chunks: list[str] = []
    total_chars = 0
    for number, page in enumerate(reader.pages, start=1):
        try:
            text = (page.extract_text() or "").strip()
        except Exception:  # noqa: BLE001 — one bad page must not lose the rest
            logger.warning("pypdf failed on page %d of %s", number, rel_path)
            text = ""
        total_chars += len(text)
        chunks.append(f"## стр. {number}\n\n{text}\n")

    if total_chars < PDF_SCAN_TEXT_CHARS:
        # No text layer at all — a scan. No sidecar to offer; the brain reads
        # the original natively, but with a hard page budget.
        return MediaPrep(instruction=_pdf_scan_instruction(rel_path), meta=meta)

    target, rel = _derived_target(vault_path, rel_path, "-text.md")
    header = f"# Извлечённый текст: {rel_path}\n\nСтраниц: {pages}\n\n"
    target.write_text(header + "\n".join(chunks), encoding="utf-8")
    return MediaPrep(
        model_rel_path=rel,
        instruction=_pdf_text_instruction(rel_path, rel),
        meta=meta,
    )


# --- Text documents -------------------------------------------------------


def _prepare_text(vault_path: Path, rel_path: str) -> MediaPrep:
    size = (vault_path / rel_path).stat().st_size
    meta = _human(size)
    if size > TEXT_CHUNK_BYTES:
        return MediaPrep(instruction=_text_chunked_instruction(rel_path), meta=meta)
    return MediaPrep(meta=meta)
