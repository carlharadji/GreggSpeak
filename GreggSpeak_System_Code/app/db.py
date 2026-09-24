import json
import sqlite3
import uuid
from datetime import datetime

from app.config import DATA_DIR, DB_PATH
from app.translation.nlp import structure_transcript


APPEND_REQUEST_PATH = DATA_DIR / "pending_append_batch.json"


CREATE_BATCHES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT UNIQUE,
    title TEXT,
    filename TEXT,
    total_pages INTEGER,
    transcript_text TEXT,
    confidence_avg REAL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    saved_location TEXT,
    upload_status TEXT,
    is_deleted INTEGER DEFAULT 0
)
"""


CREATE_PAGES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS pages (
    page_id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id TEXT,
    page_number INTEGER,
    raw_text TEXT,
    raw_transcript_text TEXT,
    edited_text TEXT,
    confidence REAL,
    image_path TEXT,
    segmentation_image_path TEXT,
    rows_detected INTEGER DEFAULT 0,
    words_detected INTEGER DEFAULT 0,
    low_confidence_words INTEGER DEFAULT 0,
    review_notes TEXT,
    is_edited INTEGER DEFAULT 0,
    is_deleted INTEGER DEFAULT 0
)
"""


def get_connection():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute(CREATE_BATCHES_TABLE_SQL)
        connection.execute(CREATE_PAGES_TABLE_SQL)
        ensure_schema(connection)
        connection.commit()


def ensure_schema(connection):
    page_columns = {
        row["name"]
        for row in connection.execute("PRAGMA table_info(pages)").fetchall()
    }
    if "segmentation_image_path" not in page_columns:
        connection.execute("ALTER TABLE pages ADD COLUMN segmentation_image_path TEXT")
    if "rows_detected" not in page_columns:
        connection.execute("ALTER TABLE pages ADD COLUMN rows_detected INTEGER DEFAULT 0")
    if "words_detected" not in page_columns:
        connection.execute("ALTER TABLE pages ADD COLUMN words_detected INTEGER DEFAULT 0")
    if "low_confidence_words" not in page_columns:
        connection.execute("ALTER TABLE pages ADD COLUMN low_confidence_words INTEGER DEFAULT 0")
    if "review_notes" not in page_columns:
        connection.execute("ALTER TABLE pages ADD COLUMN review_notes TEXT")
    if "raw_transcript_text" not in page_columns:
        connection.execute("ALTER TABLE pages ADD COLUMN raw_transcript_text TEXT")


def row_to_dict(row):
    return dict(row) if row is not None else None


def set_pending_append_batch(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()

    if batch is None:
        return {"success": False, "message": "Batch not found."}

    payload = {
        "batch_id": batch["batch_id"],
        "title": batch["title"] or "Untitled Batch",
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    APPEND_REQUEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {
        "success": True,
        "message": "Add Pages mode is ready on the LCD scanner.",
        "data": payload,
    }


def get_pending_append_batch():
    if not APPEND_REQUEST_PATH.exists():
        return None
    try:
        payload = json.loads(APPEND_REQUEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    batch_id = str(payload.get("batch_id") or "").strip()
    if not batch_id:
        return None
    return payload


def clear_pending_append_batch(batch_id=None):
    payload = get_pending_append_batch()
    if batch_id and payload and payload.get("batch_id") != batch_id:
        return
    try:
        APPEND_REQUEST_PATH.unlink()
    except FileNotFoundError:
        pass


def create_batch(title, filename, total_pages):
    init_db()

    if not str(title).strip():
        return {"success": False, "message": "Title is required."}
    try:
        total_pages = int(total_pages)
    except (TypeError, ValueError):
        return {"success": False, "message": "Total pages must be a valid integer."}

    batch_id = f"batch-{uuid.uuid4().hex[:12]}"

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO batches (
                batch_id,
                title,
                filename,
                total_pages,
                transcript_text,
                confidence_avg,
                saved_location,
                upload_status
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                str(title).strip(),
                filename,
                int(total_pages),
                "",
                0.0,
                "",
                "pending",
            ),
        )
        connection.commit()

        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()

    return {
        "success": True,
        "message": "Batch created successfully.",
        "data": row_to_dict(batch),
    }


def add_page(
    batch_id,
    page_number,
    raw_text,
    confidence,
    image_path,
    segmentation_image_path="",
    rows_detected=0,
    words_detected=0,
    low_confidence_words=0,
    review_notes="",
    raw_transcript_text="",
):
    init_db()

    try:
        page_number = int(page_number)
        confidence = float(confidence)
        rows_detected = int(rows_detected or 0)
        words_detected = int(words_detected or 0)
        low_confidence_words = int(low_confidence_words or 0)
    except (TypeError, ValueError):
        return {"success": False, "message": "Page metadata is invalid."}

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Batch not found."}

        connection.execute(
            """
            INSERT INTO pages (
                batch_id,
                page_number,
                raw_text,
                raw_transcript_text,
                edited_text,
                confidence,
                image_path,
                segmentation_image_path,
                rows_detected,
                words_detected,
                low_confidence_words,
                review_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                page_number,
                raw_text,
                raw_transcript_text,
                raw_text,
                confidence,
                image_path,
                segmentation_image_path,
                rows_detected,
                words_detected,
                low_confidence_words,
                review_notes,
            ),
        )
        connection.commit()

        page = connection.execute(
            """
            SELECT * FROM pages
            WHERE batch_id = ? AND page_number = ? AND is_deleted = 0
            ORDER BY page_id DESC
            LIMIT 1
            """,
            (batch_id, page_number),
        ).fetchone()

    return {
        "success": True,
        "message": "Page added successfully.",
        "data": row_to_dict(page),
    }


def edit_page(page_id, edited_text):
    init_db()

    try:
        page_id = int(page_id)
    except (TypeError, ValueError):
        return {"success": False, "message": "Page ID must be a valid integer."}

    with get_connection() as connection:
        page = connection.execute(
            """
            SELECT * FROM pages
            WHERE page_id = ? AND is_deleted = 0
            """,
            (page_id,),
        ).fetchone()
        if page is None:
            return {"success": False, "message": "Page not found."}

        connection.execute(
            """
            UPDATE pages
            SET edited_text = ?, is_edited = 1
            WHERE page_id = ?
            """,
            (edited_text, page_id),
        )
        connection.commit()

        updated_page = connection.execute(
            """
            SELECT * FROM pages
            WHERE page_id = ?
            """,
            (page_id,),
        ).fetchone()

    return {
        "success": True,
        "message": "Page updated successfully.",
        "data": row_to_dict(updated_page),
    }


def delete_page(page_id):
    init_db()

    try:
        page_id = int(page_id)
    except (TypeError, ValueError):
        return {"success": False, "message": "Page ID must be a valid integer."}

    with get_connection() as connection:
        page = connection.execute(
            """
            SELECT pages.*, batches.is_deleted AS batch_is_deleted
            FROM pages
            JOIN batches ON pages.batch_id = batches.batch_id
            WHERE pages.page_id = ? AND pages.is_deleted = 0
            """,
            (page_id,),
        ).fetchone()
        if page is None or page["batch_is_deleted"]:
            return {"success": False, "message": "Page not found."}

        batch_id = page["batch_id"]
        deleted_page_number = page["page_number"]

        connection.execute(
            """
            DELETE FROM pages
            WHERE page_id = ?
            """,
            (page_id,),
        )

        remaining_pages = connection.execute(
            """
            SELECT *
            FROM pages
            WHERE batch_id = ? AND is_deleted = 0
            ORDER BY page_number ASC, page_id ASC
            """,
            (batch_id,),
        ).fetchall()

        for new_page_number, remaining_page in enumerate(remaining_pages, start=1):
            if remaining_page["page_number"] != new_page_number:
                connection.execute(
                    """
                    UPDATE pages
                    SET page_number = ?
                    WHERE page_id = ?
                    """,
                    (new_page_number, remaining_page["page_id"]),
                )

        transcript_parts = [
            page_row["edited_text"] or page_row["raw_text"] or ""
            for page_row in remaining_pages
        ]
        confidence_values = [
            float(page_row["confidence"] or 0)
            for page_row in remaining_pages
        ]
        transcript_text = (
            structure_transcript("\n\n".join(part.strip() for part in transcript_parts if part.strip()))
            if remaining_pages
            else ""
        )
        confidence_avg = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else 0.0
        )
        upload_status = "processed" if remaining_pages else "pending"

        connection.execute(
            """
            UPDATE batches
            SET total_pages = ?, transcript_text = ?, confidence_avg = ?, upload_status = ?
            WHERE batch_id = ?
            """,
            (
                len(remaining_pages),
                transcript_text,
                round(confidence_avg, 2),
                upload_status,
                batch_id,
            ),
        )
        connection.commit()

    return {
        "success": True,
        "message": f"Page {deleted_page_number} permanently deleted.",
        "data": {
            "batch_id": batch_id,
            "deleted_page_number": deleted_page_number,
            "remaining_pages": len(remaining_pages),
        },
    }


def generate_transcript(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Batch not found."}

        pages = connection.execute(
            """
            SELECT * FROM pages
            WHERE batch_id = ? AND is_deleted = 0
            ORDER BY page_number ASC, page_id ASC
            """,
            (batch_id,),
        ).fetchall()
        if not pages:
            return {"success": False, "message": "No pages found for this batch."}

        transcript_parts = []
        confidence_values = []
        for page in pages:
            transcript_parts.append(page["edited_text"] or page["raw_text"] or "")
            confidence_values.append(float(page["confidence"] or 0))

        transcript_text = structure_transcript(
            "\n\n".join(part.strip() for part in transcript_parts if part.strip())
        )
        confidence_avg = (
            sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
        )

        connection.execute(
            """
            UPDATE batches
            SET total_pages = ?, transcript_text = ?, confidence_avg = ?, upload_status = ?
            WHERE batch_id = ?
            """,
            (len(pages), transcript_text, round(confidence_avg, 2), "processed", batch_id),
        )
        connection.commit()

        updated_batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()

    return {
        "success": True,
        "message": "Transcript generated successfully.",
        "data": row_to_dict(updated_batch),
    }


def update_batch_saved_location(batch_id, saved_location):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Batch not found."}

        connection.execute(
            """
            UPDATE batches
            SET saved_location = ?
            WHERE batch_id = ?
            """,
            (str(saved_location), batch_id),
        )
        connection.commit()

        updated_batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        ).fetchone()

    return {
        "success": True,
        "message": "Saved location updated successfully.",
        "data": row_to_dict(updated_batch),
    }


def get_latest_batch():
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE is_deleted = 0
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT 1
            """
        ).fetchone()

    if batch is None:
        return {"success": False, "message": "No batches found.", "data": None}

    return {
        "success": True,
        "message": "Latest batch retrieved successfully.",
        "data": row_to_dict(batch),
    }


def get_batch(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()

    if batch is None:
        return {"success": False, "message": "Batch not found.", "data": None}

    return {
        "success": True,
        "message": "Batch retrieved successfully.",
        "data": row_to_dict(batch),
    }


def get_all_batches():
    init_db()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM batches
            WHERE is_deleted = 0
            ORDER BY datetime(created_at) DESC, id DESC
            """
        ).fetchall()

    return {
        "success": True,
        "message": "Batches retrieved successfully.",
        "data": [row_to_dict(row) for row in rows],
    }


def get_deleted_batches():
    init_db()

    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT * FROM batches
            WHERE is_deleted = 1
            ORDER BY datetime(created_at) DESC, id DESC
            """
        ).fetchall()

    return {
        "success": True,
        "message": "Deleted batches retrieved successfully.",
        "data": [row_to_dict(row) for row in rows],
    }


def get_pages(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Batch not found.", "data": []}

        rows = connection.execute(
            """
            SELECT * FROM pages
            WHERE batch_id = ? AND is_deleted = 0
            ORDER BY page_number ASC, page_id ASC
            """,
            (batch_id,),
        ).fetchall()

    return {
        "success": True,
        "message": "Pages retrieved successfully.",
        "data": [row_to_dict(row) for row in rows],
    }


def soft_delete_batch(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 0
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Batch not found."}

        connection.execute(
            """
            UPDATE batches
            SET is_deleted = 1, upload_status = ?
            WHERE batch_id = ?
            """,
            ("deleted", batch_id),
        )
        connection.execute(
            """
            UPDATE pages
            SET is_deleted = 1
            WHERE batch_id = ?
            """,
            (batch_id,),
        )
        connection.commit()

    return {
        "success": True,
        "message": "Batch deleted successfully.",
        "data": {"batch_id": batch_id},
    }


def restore_batch(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 1
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Deleted batch not found."}

        connection.execute(
            """
            UPDATE batches
            SET is_deleted = 0,
                upload_status = CASE
                    WHEN transcript_text IS NOT NULL AND TRIM(transcript_text) != '' THEN 'processed'
                    ELSE 'pending'
                END
            WHERE batch_id = ?
            """,
            (batch_id,),
        )
        connection.execute(
            """
            UPDATE pages
            SET is_deleted = 0
            WHERE batch_id = ?
            """,
            (batch_id,),
        )
        connection.commit()

    return {
        "success": True,
        "message": "Batch restored successfully.",
        "data": {"batch_id": batch_id},
    }


def hard_delete_batch(batch_id):
    init_db()

    with get_connection() as connection:
        batch = connection.execute(
            """
            SELECT * FROM batches
            WHERE batch_id = ? AND is_deleted = 1
            """,
            (batch_id,),
        ).fetchone()
        if batch is None:
            return {"success": False, "message": "Deleted batch not found."}

        connection.execute(
            """
            DELETE FROM pages
            WHERE batch_id = ?
            """,
            (batch_id,),
        )
        connection.execute(
            """
            DELETE FROM batches
            WHERE batch_id = ?
            """,
            (batch_id,),
        )
        connection.commit()

    return {
        "success": True,
        "message": "Batch permanently deleted.",
        "data": {"batch_id": batch_id},
    }


init_db()
