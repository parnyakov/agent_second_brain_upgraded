"""Tests for media_prep — heavy attachments must never be handed raw to the
brain (agent-infra-backlog item 21).

All binary fixtures are GENERATED here at test-run time, never committed:
tests/fixtures/ in this repo holds text fixtures only (see
pane_salvage_incident.txt) and /vault/* plus the usual binary paths are
gitignored — a 3.8MB JPEG has no business in the code repo.
"""

import hashlib
import zlib

import pytest

from d_brain.services import media_prep
from d_brain.services.media_prep import MediaPrep, prepare_for_model

# ── fixture builders (run-time generated, nothing committed) ───────────────


def _write_image(path, size, color=(120, 90, 200), fmt=None):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color)
    # Noise so the encoder cannot compress a flat fill into a few KB —
    # the real repro file is ~3.8MB and we need the byte threshold to bite.
    pixels = img.load()
    for y in range(0, size[1], 3):
        for x in range(0, size[0], 3):
            pixels[x, y] = ((x * 7) % 256, (y * 13) % 256, ((x + y) * 3) % 256)
    img.save(path, format=fmt)
    return path


def _write_noise_image(path, size, fmt=None, **save_kwargs):
    """Incompressible image — used where the BYTE threshold, not resolution,
    has to be the thing that trips."""
    import os

    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    raw = os.urandom(size[0] * size[1] * 3)
    Image.frombytes("RGB", size, raw).save(path, format=fmt, **save_kwargs)
    return path


def _pdf_bytes(page_streams: list[bytes]) -> bytes:
    """Minimal but structurally valid PDF with one content stream per page."""
    objects: list[bytes] = []

    def add(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    n_pages = len(page_streams)
    # ids: 1 catalog, 2 pages, 3 font, then (page, content) pairs
    catalog_id, pages_id, font_id = 1, 2, 3
    objects.extend([b"", b"", b""])  # placeholders, filled below
    page_ids = []
    for stream in page_streams:
        content_id = add(
            b"<< /Length "
            + str(len(stream)).encode()
            + b" >>\nstream\n"
            + stream
            + b"\nendstream"
        )
        page_ids.append(
            add(
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents " + str(content_id).encode() + b" 0 R "
                b"/Resources << /Font << /F1 " + str(font_id).encode() + b" 0 R >> "
                b">> >>"
            )
        )
    kids = b" ".join(f"{pid} 0 R".encode() for pid in page_ids)
    objects[catalog_id - 1] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[pages_id - 1] = (
        b"<< /Type /Pages /Kids ["
        + kids
        + b"] /Count "
        + str(n_pages).encode()
        + b" >>"
    )
    objects[font_id - 1] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += str(index).encode() + b" 0 obj\n" + body + b"\nendobj\n"
    xref_at = len(out)
    out += b"xref\n0 " + str(len(objects) + 1).encode() + b"\n"
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objects) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref_at).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def _text_page(lines: list[str]) -> bytes:
    body = [b"BT", b"/F1 11 Tf", b"72 720 Td", b"14 TL"]
    for line in lines:
        escaped = line.replace("(", r"\(").replace(")", r"\)")
        body.append(b"(" + escaped.encode("latin-1") + b") Tj T*")
    body.append(b"ET")
    return b"\n".join(body)


#: A page with graphics only — no text operators at all, i.e. what a scanned
#: page looks like to pypdf: zero extractable characters.
_SCAN_PAGE = b"0.2 0.2 0.9 rg\n50 50 500 700 re\nf\n"


def _write_pdf(path, page_streams):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_pdf_bytes(page_streams))
    return path


def _vault_with(tmp_path, name, writer):
    rel = f"attachments/2026-09-04/{name}"
    writer(tmp_path / rel)
    return rel


# ── constants fixed by the plan ───────────────────────────────────────────


def test_constants_match_the_plan():
    assert media_prep.MODEL_LONG_SIDE == 1568
    assert media_prep.MAX_IMAGE_BYTES == 1_500_000
    assert media_prep.PDF_DIRECT_MAX_PAGES == 5
    assert media_prep.PDF_DIRECT_MAX_BYTES == 2_000_000
    assert media_prep.PDF_SCAN_TEXT_CHARS == 200
    assert media_prep.TEXT_CHUNK_BYTES == 256 * 1024
    assert {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"} <= media_prep.IMAGE_EXTS


def test_heic_supported_only_when_decoder_loaded():
    if media_prep.HEIF_SUPPORTED:
        assert {"heic", "heif"} <= media_prep.IMAGE_EXTS
    else:
        assert not ({"heic", "heif"} & media_prep.IMAGE_EXTS)


def test_derived_rel_path_shape():
    assert (
        media_prep.derived_rel_path(
            "attachments/2026-09-04/img-022703.jpg", "-model.jpg"
        )
        == "attachments/2026-09-04/derived/img-022703-model.jpg"
    )


# ── images: the original 30-31.08 repro ───────────────────────────────────


def test_big_document_image_is_downscaled(tmp_path):
    """The exact repro class: 4284x4284 / ~3.8MB sent as a document."""
    rel = _vault_with(
        tmp_path, "img-022703.jpg", lambda p: _write_image(p, (4284, 4284))
    )
    original = tmp_path / rel
    assert original.stat().st_size > media_prep.MAX_IMAGE_BYTES
    before = hashlib.sha256(original.read_bytes()).hexdigest()

    prep = prepare_for_model(tmp_path, rel, "document", "jpg")

    assert prep.model_rel_path == "attachments/2026-09-04/derived/img-022703-model.jpg"
    derived = tmp_path / prep.model_rel_path
    assert derived.exists()

    from PIL import Image

    with Image.open(derived) as img:
        assert max(img.size) <= media_prep.MODEL_LONG_SIDE
    assert derived.stat().st_size < original.stat().st_size

    # rule 1: the original is never mutated (owner's standing rule)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before

    assert prep.model_rel_path in prep.instruction
    assert rel in prep.instruction
    assert "ТОЛЬКО на оригинал" in prep.instruction
    assert prep.meta


def test_normal_telegram_photo_gets_no_derivative(tmp_path):
    """Matrix row 3: a plain [photo] is already ~1280px — behavior unchanged,
    no derived file created at all."""
    rel = _vault_with(
        tmp_path, "img-042213.jpg", lambda p: _write_image(p, (1280, 960))
    )
    prep = prepare_for_model(tmp_path, rel, "photo", "jpg")

    assert prep.model_rel_path is None
    assert not (tmp_path / "attachments/2026-09-04/derived").exists()
    assert "прочитай" in prep.instruction.lower()


def test_oversize_by_bytes_alone_is_downscaled(tmp_path):
    """Long side under the ceiling but the file is still huge — re-encode.

    thumbnail() is a no-op here (nothing to shrink by dimension), so the ONLY
    saving comes from the re-encode. The size assertion below is the same one
    the sibling test makes, and it is the point of the whole exercise: a
    derivative the prompt calls "the smaller copy" has to actually be one.
    """
    rel = _vault_with(
        tmp_path,
        "wide.jpg",
        lambda p: _write_noise_image(p, (1500, 1400), fmt="JPEG", quality=100),
    )
    original = tmp_path / rel
    assert original.stat().st_size > media_prep.MAX_IMAGE_BYTES

    prep = prepare_for_model(tmp_path, rel, "document", "jpg")

    assert prep.model_rel_path and prep.model_rel_path.endswith("-model.jpg")
    derived = tmp_path / prep.model_rel_path
    assert derived.exists()
    assert derived.stat().st_size < original.stat().st_size

    from PIL import Image

    with Image.open(derived) as img:
        # dimensions untouched — the byte threshold is what tripped
        assert img.size == (1500, 1400)


def test_derivative_that_is_not_smaller_is_discarded(tmp_path):
    """An incompressible image under the resolution ceiling re-encodes to the
    same size (or bigger). Keeping it would make the prompt lie — "read this
    smaller copy, don't open the heavy original" — and waste disk, so it must
    fall back to the plain use-the-original path with nothing left behind."""
    rel = _vault_with(
        tmp_path, "noise.png", lambda p: _write_noise_image(p, (1500, 1400))
    )
    original = tmp_path / rel
    assert original.stat().st_size > media_prep.MAX_IMAGE_BYTES
    before = hashlib.sha256(original.read_bytes()).hexdigest()

    prep = prepare_for_model(tmp_path, rel, "document", "png")

    assert prep.model_rel_path is None
    assert "прочитай" in prep.instruction.lower()
    # no orphan derivative left on disk
    derived_dir = tmp_path / "attachments/2026-09-04/derived"
    assert not derived_dir.exists() or not list(derived_dir.iterdir())
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before


def test_broken_image_degrades_to_original(tmp_path):
    rel = "attachments/2026-09-04/img-broken.jpg"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(b"this is definitely not a jpeg")

    prep = prepare_for_model(tmp_path, rel, "document", "jpg")

    assert prep == MediaPrep()  # exactly today's behavior, no exception


def test_missing_file_degrades_to_original(tmp_path):
    prep = prepare_for_model(tmp_path, "attachments/nope/gone.jpg", "document", "jpg")
    assert prep == MediaPrep()


# ── PDF ───────────────────────────────────────────────────────────────────


def test_small_pdf_is_read_directly(tmp_path):
    rel = _vault_with(
        tmp_path,
        "small.pdf",
        lambda p: _write_pdf(p, [_text_page([f"page {i} line"]) for i in range(1, 4)]),
    )
    prep = prepare_for_model(tmp_path, rel, "document", "pdf")

    assert prep.model_rel_path is None
    assert "3 стр." in prep.meta
    assert "напрямую" in prep.instruction


def test_big_text_pdf_gets_a_sidecar(tmp_path):
    pages = [
        _text_page([f"Strategy page {i}", "revenue plan for the quarter"] * 12)
        for i in range(1, 13)
    ]
    rel = _vault_with(tmp_path, "report.pdf", lambda p: _write_pdf(p, pages))

    prep = prepare_for_model(tmp_path, rel, "document", "pdf")

    assert prep.model_rel_path == "attachments/2026-09-04/derived/report-text.md"
    sidecar = (tmp_path / prep.model_rel_path).read_text(encoding="utf-8")
    assert "## стр. 1" in sidecar and "## стр. 12" in sidecar
    assert "Strategy page 7" in sidecar
    assert "12 стр." in prep.meta
    assert prep.model_rel_path in prep.instruction
    assert "5 страниц за раз" in prep.instruction


def test_scan_pdf_gets_page_limited_native_read(tmp_path):
    rel = _vault_with(
        tmp_path, "scan.pdf", lambda p: _write_pdf(p, [_SCAN_PAGE] * 20)
    )
    prep = prepare_for_model(tmp_path, rel, "document", "pdf")

    assert prep.model_rel_path is None  # nothing to extract from a scan
    assert "скан" in prep.instruction
    assert "1-5" in prep.instruction
    assert "20 стр." in prep.meta


def test_broken_pdf_degrades_to_original(tmp_path):
    rel = "attachments/2026-09-04/broken.pdf"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(b"%PDF-1.4 not really")
    assert prepare_for_model(tmp_path, rel, "document", "pdf") == MediaPrep()


# ── honest bounded turns: video / audio / binary / big text ───────────────


@pytest.mark.parametrize(
    "kind,ext",
    [
        ("video", "mp4"),
        ("animation", "mp4"),
        ("video_note", "mp4"),
        ("document", "mov"),
    ],
)
def test_video_is_never_read(tmp_path, kind, ext):
    rel = f"attachments/2026-09-04/clip.{ext}"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(b"\x00\x01" * 5000)

    prep = prepare_for_model(tmp_path, rel, kind, ext)

    assert prep.model_rel_path is None
    assert "НЕ читай" in prep.instruction
    assert rel in prep.instruction
    # no invented capability
    assert "обработать не могу" in prep.instruction


def test_audio_file_is_not_transcribed(tmp_path):
    rel = "attachments/2026-09-04/track.mp3"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(b"ID3" + b"\x00" * 4000)

    prep = prepare_for_model(tmp_path, rel, "audio", "mp3")

    assert prep.model_rel_path is None
    assert "не транскрибирую" in prep.instruction
    assert "НЕ читай" in prep.instruction


@pytest.mark.parametrize("ext", ["zip", "docx", "xlsx", "bin"])
def test_unknown_binary_is_not_read(tmp_path, ext):
    rel = f"attachments/2026-09-04/blob.{ext}"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_bytes(zlib.compress(b"x" * 100))

    prep = prepare_for_model(tmp_path, rel, "document", ext)

    assert prep.model_rel_path is None
    assert "НЕ читай бинарный файл" in prep.instruction


def test_big_text_document_is_read_in_chunks(tmp_path):
    rel = "attachments/2026-09-04/log.txt"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text("line of log output\n" * 40_000, encoding="utf-8")

    prep = prepare_for_model(tmp_path, rel, "document", "txt")

    assert prep.model_rel_path is None
    assert "НЕ читай целиком" in prep.instruction
    assert "500" in prep.instruction


def test_small_text_document_is_read_whole(tmp_path):
    rel = "attachments/2026-09-04/note.md"
    (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / rel).write_text("# hi\n", encoding="utf-8")

    prep = prepare_for_model(tmp_path, rel, "document", "md")

    assert prep.model_rel_path is None
    assert prep.instruction == media_prep.DEFAULT_INSTRUCTION
