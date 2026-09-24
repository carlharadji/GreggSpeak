import hmac
import os
import re
import textwrap
import zipfile
from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path
from xml.sax.saxutils import escape as escape_xml

from flask import Flask, Response, abort, flash, jsonify, redirect, render_template, request, send_file, session, url_for

from app.db import (
    add_page,
    create_batch,
    delete_page,
    edit_page,
    generate_transcript,
    get_batch,
    get_all_batches,
    get_deleted_batches,
    get_latest_batch,
    get_pages,
    hard_delete_batch,
    init_db,
    restore_batch,
    set_pending_append_batch,
    soft_delete_batch,
)
from app.config import DATA_DIR, DB_PATH, EXPORTS_DIR, STATIC_DIR, TEMPLATES_DIR, WEB_HOST, WEB_PORT
from app.pipeline.runner import PageInput, recognize_and_store
from app.preprocessing.input_source import capture_camera_image
from app.tts.speaker import speak_text
from app.utils.network import get_access_urls, get_primary_access_url, print_access_urls


app = Flask(
    __name__,
    template_folder=str(TEMPLATES_DIR),
    static_folder=str(STATIC_DIR),
)
app.secret_key = os.environ.get("GREGGSPEAK_SECRET_KEY", "").strip()
if not app.secret_key:
    raise RuntimeError("Set GREGGSPEAK_SECRET_KEY before starting the web app.")
init_db()
DEFAULT_HOST = WEB_HOST
DEFAULT_PORT = WEB_PORT
WEB_USERNAME = os.environ.get("GREGGSPEAK_WEB_USERNAME", "admin")
WEB_PASSWORD = os.environ.get("GREGGSPEAK_WEB_PASSWORD", "").strip()
if not WEB_PASSWORD:
    raise RuntimeError("Set GREGGSPEAK_WEB_PASSWORD before starting the web app.")
PUBLIC_ENDPOINTS = {"login_page", "logout_page", "static", "access_qr_png"}


@app.context_processor
def inject_network_access():
    return {
        "access_urls": get_access_urls(DEFAULT_PORT),
        "primary_access_url": get_primary_access_url(DEFAULT_PORT),
        "is_authenticated": bool(session.get("greggspeak_authenticated")),
        "current_username": session.get("greggspeak_username", "User"),
    }


@app.before_request
def require_login():
    if request.endpoint in PUBLIC_ENDPOINTS or request.path.startswith("/static/"):
        return None

    if session.get("greggspeak_authenticated"):
        return None

    if request.path.startswith("/api/"):
        return jsonify({"success": False, "message": "Login required."}), 401

    return redirect(url_for("login_page", next=safe_next_url(request.full_path)))


def safe_next_url(value):
    value = str(value or "").strip()
    if not value or not value.startswith("/") or value.startswith("//"):
        return url_for("user_dashboard")
    if value.endswith("?"):
        value = value[:-1]
    return value


def build_response(result, success_code=200, error_code=400):
    status_code = success_code if result.get("success") else error_code
    return jsonify(result), status_code


def sanitize_filename(value):
    cleaned = "".join(char if char.isalnum() else "_" for char in (value or "transcript"))
    cleaned = "_".join(filter(None, cleaned.split("_")))
    return cleaned or "transcript"


def escape_pdf_text(value):
    return (
        str(value)
        .replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
    )


def build_pdf_bytes(title, lines):
    page_width = 612
    page_height = 792
    left_margin = 50
    top_margin = 740
    line_height = 16
    usable_lines = 40

    wrapped_lines = []
    for line in lines:
        parts = textwrap.wrap(str(line), width=88) or [""]
        wrapped_lines.extend(parts)

    if not wrapped_lines:
        wrapped_lines = ["Transcript not available."]

    pages = []
    for index in range(0, len(wrapped_lines), usable_lines):
        chunk = wrapped_lines[index:index + usable_lines]
        stream_lines = ["BT", "/F1 12 Tf"]
        y_position = top_margin

        for chunk_line in chunk:
            stream_lines.append(f"1 0 0 1 {left_margin} {y_position} Tm ({escape_pdf_text(chunk_line)}) Tj")
            y_position -= line_height

        stream_lines.append("ET")
        stream = "\n".join(stream_lines).encode("latin-1", errors="replace")
        pages.append(stream)

    objects = []
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")

    kids = " ".join(f"{4 + (page_index * 2)} 0 R" for page_index in range(len(pages)))
    objects.append(f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode("latin-1"))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    for page_index, stream in enumerate(pages):
        content_object_id = 5 + (page_index * 2)
        page_object = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object_id} 0 R >>"
        ).encode("latin-1")
        content_object = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") +
            stream +
            b"\nendstream"
        )
        objects.append(page_object)
        objects.append(content_object)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]

    for object_index, object_body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_index} 0 obj\n".encode("latin-1"))
        pdf.extend(object_body)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)


TSN_FIELDS = {
    "header_line_1": "",
    "header_line_2": "",
    "header_line_3": "",
    "header_line_4": "",
    "email": "",
    "mobile": "",
    "complainant": "",
    "criminal_case_no": "0000",
    "case_for": "",
    "accused": "",
    "proceeding": "",
    "hearing_date": "",
    "hearing_time": "",
    "hearing_period": "afternoon",
    "place": "",
    "judge_name": "",
    "public_prosecutor": "",
    "defense_counsel": "",
    "court_interpreter": "",
    "court_stenographer": "",
    "order_text": "",
    "certifier_name": "",
}
TSN_PAGE_WIDTH = 88
TSN_RULE = "x" + ("-" * (TSN_PAGE_WIDTH - 2)) + "x"
TSN_BODY_INDENT = "        "
TSN_BODY_LINES_PER_PAGE = 45


def default_tsn_fields(batch=None):
    values = dict(TSN_FIELDS)
    if batch:
        values["criminal_case_no"] = values["criminal_case_no"] or str(batch.get("batch_id") or "0000")
        values["certifier_name"] = values["court_stenographer"]
    return values


def get_tsn_form_data(batch=None):
    defaults = default_tsn_fields(batch)
    data = {}
    for key, default in defaults.items():
        value = request.form.get(key, default)
        data[key] = str(value).strip()
    data["hearing_date"] = format_tsn_date(data.get("hearing_date"))
    data["hearing_time"] = format_tsn_time(data.get("hearing_time"))
    return data


def format_tsn_date(value):
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        return datetime.strptime(value, "%Y-%m-%d").strftime("%B %d, %Y")
    except ValueError:
        return value


def format_tsn_time(value):
    value = str(value or "").strip()
    if not value:
        return ""
    try:
        parsed = datetime.strptime(value, "%H:%M")
        return parsed.strftime("%I:%M").lstrip("0")
    except ValueError:
        return value


def blank_or_value(value, width=28):
    value = str(value or "").strip()
    if value:
        return value
    return "_" * width


def build_tsn_pdf_bytes(batch, transcript_text, fields):
    pages = build_tsn_document_pages(batch, transcript_text, fields)
    return build_pdf_pages_bytes(pages)


def build_tsn_docx_bytes(batch, transcript_text, fields):
    pages = build_tsn_document_pages(batch, transcript_text, fields)
    return build_docx_bytes(pages)


def build_tsn_document_pages(batch, transcript_text, fields):
    del batch
    front_page = build_tsn_front_page(fields)
    transcript_pages = build_tsn_transcript_pages(transcript_text, fields)
    body_pages = transcript_pages + [build_tsn_certification_page(fields)]
    total_pages = 1 + len(body_pages)

    pages = [front_page]
    for page_index, page_lines in enumerate(body_pages, start=2):
        page = build_tsn_page_header(fields, page_index, total_pages) + [""] + page_lines
        pages.append(page)
    return pages


def build_tsn_front_page(fields):
    complainant = blank_or_value(fields.get("complainant"), 30)
    accused = blank_or_value(fields.get("accused"), 30)
    complainant_line = f"{complainant},"
    accused_line = f"{accused},"
    party_column_width = max(len(complainant_line), len(accused_line), 34)
    case_no = blank_or_value(fields.get("criminal_case_no"), 8)
    case_for = blank_or_value(fields.get("case_for"), 26)
    proceeding = blank_or_value(fields.get("proceeding"), 18)
    hearing_date = blank_or_value(fields.get("hearing_date"), 14)
    hearing_time = blank_or_value(fields.get("hearing_time"), 8)
    hearing_period = blank_or_value(fields.get("hearing_period"), 12)
    place = blank_or_value(fields.get("place"), 24)
    judge_name = blank_or_value(fields.get("judge_name"), 24)

    lines = [
        center_pdf_line(blank_or_value(fields.get("header_line_1"), 36)),
        center_pdf_line(blank_or_value(fields.get("header_line_2"), 36)),
        center_pdf_line(blank_or_value(fields.get("header_line_3"), 36)),
        center_pdf_line(blank_or_value(fields.get("header_line_4"), 36)),
        center_pdf_line(f"Email Add: {blank_or_value(fields.get('email'), 28)}"),
        center_pdf_line(f"Mobile Number: {blank_or_value(fields.get('mobile'), 16)}"),
        "",
        left_right(complainant_line, f"Criminal Case No. {case_no}"),
        left_right("- versus -".center(party_column_width), f"For: {case_for}"),
        "",
        accused_line,
        indent_text("Accused.", 12),
        TSN_RULE,
        "",
        center_pdf_line("T R A N S C R I P T"),
        "",
        (
            "of the stenographic notes taken down by the undersigned during the "
            f"{proceeding} in the above-entitled case held on {hearing_date} "
            f"at {hearing_time} o'clock in the {hearing_period} at the {place} "
            f"before the {judge_name}, Presiding Judge."
        ),
        "",
        "A P P E A R A N C E S:",
        "",
        *center_role_pair(fields.get("public_prosecutor"), "Public Prosecutor"),
        *center_role_pair(fields.get("defense_counsel"), "Counsel for the Accused"),
        "P R E S E N T:",
        "",
        *center_role_pair(fields.get("court_interpreter"), "Court Interpreter"),
        *center_role_pair(fields.get("court_stenographer"), "Court Stenographer"),
        TSN_RULE,
    ]
    return wrap_pdf_lines(lines, width=TSN_PAGE_WIDTH)


def build_tsn_page_header(fields, page_number, total_pages):
    case_no = blank_or_value(fields.get("criminal_case_no"), 8)
    complainant = blank_or_value(fields.get("complainant"), 19)
    accused = blank_or_value(fields.get("accused"), 19)
    case_for = blank_or_value(fields.get("case_for"), 26)
    proceeding = str(fields.get("proceeding") or "").strip() or "Trial"
    date_text = blank_or_value(fields.get("hearing_date"), 14)
    return [
        "TSN",
        f"Criminal Case No. {case_no}",
        f"{complainant} vs. {accused}",
        f"For: {case_for}",
        proceeding,
        left_right(f"Page {page_number} of {total_pages}", date_text),
        TSN_RULE,
    ]


def build_tsn_transcript_pages(transcript_text, fields):
    lines = format_tsn_transcript_body(str(transcript_text or "Transcript not generated yet.").splitlines())
    order_text = str(fields.get("order_text") or "").strip()
    if order_text:
        lines.extend(["", "COURT:"])
        lines.extend(wrap_pdf_lines([f"Order. {order_text}. So Ordered."], width=TSN_PAGE_WIDTH - len(TSN_BODY_INDENT), indent=TSN_BODY_INDENT))
    lines.append(TSN_RULE)

    if not lines:
        lines = ["Transcript not generated yet."]
    return [lines[index:index + TSN_BODY_LINES_PER_PAGE] for index in range(0, len(lines), TSN_BODY_LINES_PER_PAGE)]


def build_tsn_certification_page(fields):
    certifier = blank_or_value(fields.get("certifier_name") or fields.get("court_stenographer"), 20)
    return [
        TSN_RULE,
        "",
        "I HEREBY CERTIFY THAT THIS IS TRUE AND CORRECT TO THE BEST OF MY KNOWLEDGE AND BELIEF.",
        "",
        "",
        "",
        f"___    {certifier}",
        "         Stenographer",
    ]


def get_line_text(line):
    if isinstance(line, dict):
        return str(line.get("text", ""))
    return str(line)


def center_pdf_line(text, width=TSN_PAGE_WIDTH):
    return str(text).center(width)


def right_text(text, width=TSN_PAGE_WIDTH):
    return str(text).rjust(width)


def indent_text(text, spaces):
    return (" " * max(0, int(spaces))) + str(text)


def center_role_pair(name, role):
    return [
        center_pdf_line(blank_or_value(name, 28)),
        center_pdf_line(role),
        "",
    ]


def left_right(left, right, width=TSN_PAGE_WIDTH):
    left = str(left or "")
    right = str(right or "")
    if len(left) + len(right) + 1 >= width:
        return f"{left} {right}".strip()
    return left + (" " * (width - len(left) - len(right))) + right


def wrap_pdf_lines(lines, width=TSN_PAGE_WIDTH, indent=""):
    wrapped = []
    for line in lines:
        raw_line = str(line)
        if not raw_line.strip():
            wrapped.append("")
            continue
        if len(raw_line) <= width:
            wrapped.append(raw_line if not indent else f"{indent}{raw_line}")
            continue
        wrapped.extend(
            textwrap.wrap(
                raw_line,
                width=max(20, width),
                initial_indent=indent,
                subsequent_indent=indent,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return wrapped


def format_tsn_transcript_body(lines):
    formatted = []
    speaker_pattern = re.compile(r"^[A-Z][A-Z ]{1,32}:$")
    for line in lines:
        stripped = str(line or "").strip()
        if not stripped:
            formatted.append("")
            continue
        if speaker_pattern.match(stripped):
            if formatted and formatted[-1] != "":
                formatted.append("")
            formatted.append(stripped)
            continue
        formatted.extend(
            wrap_pdf_lines(
                [stripped],
                width=TSN_PAGE_WIDTH - len(TSN_BODY_INDENT),
                indent=TSN_BODY_INDENT,
            )
        )
    return formatted


def build_pdf_pages_bytes(pages):
    page_width = 612
    page_height = 792
    left_margin = 46
    top_margin = 748
    line_height = 13
    font_size = 10
    max_lines = 54

    normalized_pages = []
    for page in pages:
        page_lines = list(page or [""])
        for index in range(0, len(page_lines), max_lines):
            normalized_pages.append(page_lines[index:index + max_lines])

    if not normalized_pages:
        normalized_pages = [["Transcript not available."]]

    streams = []
    for page_lines in normalized_pages:
        stream_lines = ["BT", f"/F1 {font_size} Tf"]
        y_position = top_margin
        for line in page_lines:
            line_text = get_line_text(line)
            stream_lines.append(f"1 0 0 1 {left_margin} {y_position} Tm ({escape_pdf_text(line_text)}) Tj")
            y_position -= line_height
        stream_lines.append("ET")
        streams.append("\n".join(stream_lines).encode("latin-1", errors="replace"))

    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Count {len(streams)} /Kids [{' '.join(f'{4 + (i * 2)} 0 R' for i in range(len(streams)))}] >>".encode("latin-1"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
    ]

    for page_index, stream in enumerate(streams):
        content_object_id = 5 + (page_index * 2)
        page_object = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page_width} {page_height}] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {content_object_id} 0 R >>"
        ).encode("latin-1")
        content_object = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("latin-1") +
            stream +
            b"\nendstream"
        )
        objects.append(page_object)
        objects.append(content_object)

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for object_index, object_body in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{object_index} 0 obj\n".encode("latin-1"))
        pdf.extend(object_body)
        pdf.extend(b"\nendobj\n")

    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))
    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_start}\n%%EOF"
        ).encode("latin-1")
    )
    return bytes(pdf)


def build_docx_bytes(pages):
    document_xml = build_docx_document_xml(pages)
    styles_xml = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>
        <w:sz w:val="20"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
</w:styles>"""
    content_types = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>"""
    root_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>"""
    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml)
        archive.writestr("word/styles.xml", styles_xml)
        archive.writestr("word/_rels/document.xml.rels", document_rels)
    return buffer.getvalue()


def build_docx_document_xml(pages):
    body_parts = []
    for page_index, page in enumerate(pages):
        for line in page:
            body_parts.append(docx_paragraph(line))
        if page_index < len(pages) - 1:
            body_parts.append(docx_page_break())

    body_parts.append(
        """
<w:sectPr>
  <w:pgSz w:w="12240" w:h="15840"/>
  <w:pgMar w:top="720" w:right="720" w:bottom="720" w:left="720" w:header="360" w:footer="360" w:gutter="0"/>
</w:sectPr>"""
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {''.join(body_parts)}
  </w:body>
</w:document>"""


def docx_paragraph(text):
    safe_text = escape_xml(get_line_text(text))
    return f"""
<w:p>
  <w:pPr>
    <w:spacing w:after="0" w:line="240" w:lineRule="auto"/>
  </w:pPr>
  <w:r>
    <w:rPr>
      <w:rFonts w:ascii="Courier New" w:hAnsi="Courier New"/>
      <w:sz w:val="20"/>
    </w:rPr>
    <w:t xml:space="preserve">{safe_text}</w:t>
  </w:r>
</w:p>"""


def docx_page_break():
    return """
<w:p>
  <w:r>
    <w:br w:type="page"/>
  </w:r>
</w:p>"""


def get_workspace_context():
    batches_result = get_all_batches()
    deleted_batches_result = get_deleted_batches()
    latest_result = get_latest_batch()
    return {
        "batches": batches_result.get("data", []),
        "deleted_batches": deleted_batches_result.get("data", []),
        "latest_batch": latest_result.get("data"),
    }


@app.route("/access/qr.png")
def access_qr_png():
    url = get_primary_access_url(DEFAULT_PORT)
    try:
        import qrcode  # type: ignore
    except Exception:
        return Response(_missing_qr_svg(url), mimetype="image/svg+xml", status=503)

    image = qrcode.make(url)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return Response(buffer.getvalue(), mimetype="image/png")


def _missing_qr_svg(url):
    escaped_url = escape(url)
    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 180 180">
<rect width="180" height="180" fill="#ffffff"/>
<rect x="10" y="10" width="160" height="160" fill="#f4f4f4" stroke="#333333"/>
<text x="90" y="72" text-anchor="middle" font-family="Arial" font-size="13" fill="#111111">QR unavailable</text>
<text x="90" y="94" text-anchor="middle" font-family="Arial" font-size="10" fill="#333333">Install qrcode[pil]</text>
<text x="90" y="118" text-anchor="middle" font-family="Arial" font-size="9" fill="#333333">{escaped_url}</text>
</svg>"""


@app.route("/artifact/<path:artifact_path>")
def page_artifact(artifact_path):
    candidate = Path(artifact_path)
    if not candidate.is_absolute():
        candidate = DATA_DIR / artifact_path

    try:
        resolved = candidate.resolve()
        resolved.relative_to(DATA_DIR.resolve())
    except Exception:
        abort(404)

    if not resolved.is_file():
        abort(404)

    return send_file(resolved)


@app.route("/")
def index():
    return redirect(url_for("user_dashboard"))


@app.route("/login", methods=["GET", "POST"])
def login_page():
    next_url = safe_next_url(request.values.get("next"))
    if session.get("greggspeak_authenticated"):
        return redirect(url_for("user_dashboard"))

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        username_ok = hmac.compare_digest(username, WEB_USERNAME)
        password_ok = hmac.compare_digest(password, WEB_PASSWORD)

        if username_ok and password_ok:
            session.clear()
            session["greggspeak_authenticated"] = True
            session["greggspeak_username"] = username
            flash("Welcome back.", "success")
            return redirect(url_for("user_dashboard"))

        flash("Invalid username or password.", "error")

    return render_template("login.html", next_url=next_url)


@app.route("/admin")
def admin_dashboard():
    return redirect(url_for("user_dashboard"))


def render_stenographer_dashboard(page_mode="home"):
    return render_template(
        "user.html",
        page_mode=page_mode,
        **get_workspace_context(),
    )


@app.route("/stenographer")
def stenographer_dashboard():
    return render_stenographer_dashboard(page_mode="home")


@app.route("/admin/batch/<batch_id>")
def admin_batch_detail(batch_id):
    return redirect(url_for("user_batch_detail", batch_id=batch_id))


@app.route("/user")
def user_dashboard():
    return render_stenographer_dashboard(page_mode="home")

@app.route("/recognize/camera", methods=["POST"])
def recognize_camera():
    image = capture_camera_image()
    if image is None:
        flash("Camera unavailable. Use image upload or check the Raspberry Pi camera setup.", "error")
        return redirect(url_for("user_dashboard"))

    return recognize_pages_into_workspace([
        PageInput(source_name="camera_capture", image=image, image_path="camera_capture", apply_fixed_crop=False)
    ])


def recognize_pages_into_workspace(pages):
    title = (request.form.get("title") or "").strip()
    use_uniform_crop = bool(request.form.get("uniform_crop"))
    should_speak = bool(request.form.get("speak"))

    try:
        result = recognize_and_store(
            pages=pages,
            title=title,
            use_uniform_crop=use_uniform_crop,
            laptop_compatible=True,
        )
    except Exception as exc:
        flash(f"Recognition failed: {exc}", "error")
        return redirect(url_for("user_dashboard"))

    if not result.get("success"):
        flash(result.get("message", "Recognition failed."), "error")
        return redirect(url_for("user_dashboard"))

    batch = result.get("batch") or {}
    if should_speak:
        speak_text(batch.get("transcript_text") or "")

    flash("Transcript batch saved.", "success")
    return redirect(url_for("user_batch_detail", batch_id=result["batch_id"]))



@app.route("/records")
def records_page():
    return render_stenographer_dashboard(page_mode="records")


@app.route("/trash")
def trash_page():
    return render_stenographer_dashboard(page_mode="trash")


@app.route("/account")
def account_page():
    context = get_workspace_context()
    return render_template(
        "account.html",
        latest_batch=context.get("latest_batch"),
        total_batches=len(context.get("batches", [])),
        deleted_batches=len(context.get("deleted_batches", [])),
    )


@app.route("/settings")
def settings_page():
    context = get_workspace_context()
    latest_batch = context.get("latest_batch")
    return render_template(
        "settings.html",
        total_batches=len(context.get("batches", [])),
        deleted_batches=len(context.get("deleted_batches", [])),
        latest_batch_id=(latest_batch or {}).get("batch_id"),
        web_url=get_primary_access_url(DEFAULT_PORT),
        database_path=str(DB_PATH),
    )


@app.route("/logout")
def logout_page():
    session.clear()
    flash("Workspace session closed.", "success")
    return redirect(url_for("login_page"))


@app.route("/user/batch/<batch_id>")
def user_batch_detail(batch_id):
    batch_result = get_batch(batch_id)
    pages_result = get_pages(batch_id)
    if not batch_result.get("success"):
        flash(batch_result.get("message", "Batch not found."), "error")
        return redirect(url_for("user_dashboard"))

    return render_template(
        "user_batch.html",
        batch=batch_result.get("data"),
        pages=pages_result.get("data", []),
        tsn_fields=default_tsn_fields(batch_result.get("data") or {}),
    )


@app.route("/batch/<batch_id>/add-pages", methods=["POST"])
def add_pages_to_batch(batch_id):
    result = set_pending_append_batch(batch_id)
    flash(
        result.get("message", "Add Pages mode is ready.") if result.get("success") else result.get("message"),
        "success" if result.get("success") else "error",
    )
    return redirect(url_for("user_batch_detail", batch_id=batch_id))


@app.route("/batch/<batch_id>/download", methods=["GET", "POST"])
def download_batch_pdf(batch_id):
    batch_result = get_batch(batch_id)
    if not batch_result.get("success"):
        flash(batch_result.get("message", "Batch not found."), "error")
        return redirect(url_for("admin_dashboard"))

    batch = batch_result.get("data") or {}
    transcript_text = batch.get("transcript_text") or "Transcript not generated yet."
    if request.method == "POST":
        fields = get_tsn_form_data(batch)
        export_format = request.form.get("export_format", "pdf").lower()
        if export_format in {"word", "docx"}:
            docx_bytes = build_tsn_docx_bytes(batch, transcript_text, fields)
            filename = f"{sanitize_filename(batch.get('title') or batch_id)}_tsn.docx"
            return Response(
                docx_bytes,
                mimetype="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )

        pdf_bytes = build_tsn_pdf_bytes(batch, transcript_text, fields)
        filename = f"{sanitize_filename(batch.get('title') or batch_id)}_tsn.pdf"
        return Response(
            pdf_bytes,
            mimetype="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    lines = [
        "GreggSpeak Transcript Export",
        "",
        f"Title: {batch.get('title') or 'Untitled Batch'}",
        f"Batch ID: {batch.get('batch_id') or '-'}",
        f"Created At: {batch.get('created_at') or '-'}",
        f"Total Pages: {batch.get('total_pages') or 0}",
        f"Confidence Average: {batch.get('confidence_avg') or 0}",
        "",
        "Combined Transcript",
        "",
    ]
    lines.extend(str(transcript_text).splitlines())

    pdf_bytes = build_pdf_bytes(batch.get("title") or "Transcript", lines)
    filename = f"{sanitize_filename(batch.get('title') or batch_id)}.pdf"
    return Response(
        pdf_bytes,
        mimetype="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.route("/admin/page/<int:page_id>/edit", methods=["POST"])
def admin_edit_page(page_id):
    return user_edit_page(page_id)


@app.route("/admin/batch/<batch_id>/delete", methods=["POST"])
def admin_delete_batch(batch_id):
    return delete_batch(batch_id)


@app.route("/admin/batch/<batch_id>/restore", methods=["POST"])
def admin_restore_batch(batch_id):
    return restore_batch_route(batch_id)


@app.route("/admin/batch/<batch_id>/purge", methods=["POST"])
def admin_purge_batch(batch_id):
    return permanently_delete_batch(batch_id)


def save_page_update(page_id, batch_id, edited_text):
    edited_text = request.form.get("edited_text", "")

    edit_result = edit_page(page_id, edited_text)
    if edit_result.get("success") and batch_id:
        generate_transcript(batch_id)
        flash("Page updated successfully.", "success")
    else:
        flash(edit_result.get("message", "Could not update page."), "error")

    return redirect(url_for("user_batch_detail", batch_id=batch_id))


def delete_batch(batch_id):
    result = soft_delete_batch(batch_id)
    flash(
        result.get("message", "Batch deleted.") if result.get("success") else result.get("message"),
        "success" if result.get("success") else "error",
    )
    return redirect(url_for("records_page"))


def restore_batch_route(batch_id):
    result = restore_batch(batch_id)
    flash(
        result.get("message", "Batch restored.") if result.get("success") else result.get("message"),
        "success" if result.get("success") else "error",
    )
    return redirect(url_for("trash_page"))


def permanently_delete_batch(batch_id):
    result = hard_delete_batch(batch_id)
    flash(
        result.get("message", "Batch permanently deleted.") if result.get("success") else result.get("message"),
        "success" if result.get("success") else "error",
    )
    return redirect(url_for("trash_page"))


@app.route("/user/page/<int:page_id>/edit", methods=["POST"])
def user_edit_page(page_id):
    batch_id = request.form.get("batch_id", "")
    edited_text = request.form.get("edited_text", "")

    return save_page_update(page_id, batch_id, edited_text)


@app.route("/user/page/<int:page_id>/delete", methods=["POST"])
def user_delete_page(page_id):
    fallback_batch_id = request.form.get("batch_id", "")
    result = delete_page(page_id)
    batch_id = (result.get("data") or {}).get("batch_id") or fallback_batch_id
    flash(
        result.get("message", "Page deleted.") if result.get("success") else result.get("message"),
        "success" if result.get("success") else "error",
    )
    if batch_id:
        return redirect(url_for("user_batch_detail", batch_id=batch_id))
    return redirect(url_for("records_page"))


@app.route("/batch/<batch_id>/delete", methods=["POST"])
def unified_delete_batch(batch_id):
    return delete_batch(batch_id)


@app.route("/batch/<batch_id>/restore", methods=["POST"])
def unified_restore_batch(batch_id):
    return restore_batch_route(batch_id)


@app.route("/batch/<batch_id>/purge", methods=["POST"])
def unified_purge_batch(batch_id):
    return permanently_delete_batch(batch_id)


@app.route("/api/batch/create", methods=["POST"])
def create_batch_route():
    payload = request.get_json(silent=True) or {}
    result = create_batch(
        title=payload.get("title", ""),
        filename=payload.get("filename", ""),
        total_pages=payload.get("total_pages", 0),
    )
    return build_response(result, success_code=201)


@app.route("/api/pages/add", methods=["POST"])
def add_page_route():
    payload = request.get_json(silent=True) or {}
    result = add_page(
        batch_id=payload.get("batch_id", ""),
        page_number=payload.get("page_number", 0),
        raw_text=payload.get("raw_text", ""),
        confidence=payload.get("confidence", 0),
        image_path=payload.get("image_path", ""),
        segmentation_image_path=payload.get("segmentation_image_path", ""),
        rows_detected=payload.get("rows_detected", 0),
        words_detected=payload.get("words_detected", 0),
        low_confidence_words=payload.get("low_confidence_words", 0),
        review_notes=payload.get("review_notes", ""),
    )
    return build_response(result, success_code=201)


@app.route("/api/page/edit", methods=["PUT"])
def edit_page_route():
    payload = request.get_json(silent=True) or {}
    result = edit_page(
        page_id=payload.get("page_id", 0),
        edited_text=payload.get("edited_text", ""),
    )
    return build_response(result)


@app.route("/api/batch/latest", methods=["GET"])
def get_latest_batch_route():
    result = get_latest_batch()
    return build_response(result, error_code=404)


@app.route("/api/batches/all", methods=["GET"])
def get_all_batches_route():
    result = get_all_batches()
    return build_response(result)


@app.route("/api/pages/<batch_id>", methods=["GET"])
def get_pages_route(batch_id):
    result = get_pages(batch_id)
    return build_response(result, error_code=404)


@app.route("/api/batch/delete", methods=["DELETE"])
def delete_batch_route():
    payload = request.get_json(silent=True) or {}
    result = soft_delete_batch(payload.get("batch_id", ""))
    return build_response(result, error_code=404)


if __name__ == "__main__":
    print_access_urls(DEFAULT_PORT)
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False, use_reloader=False)
