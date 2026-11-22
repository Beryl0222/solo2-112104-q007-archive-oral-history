"""Data models and source-record loading for the oral-history indexer.

The source material is never rewritten.  Every derived object keeps a stable
reference back to the record, version and character span it came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


class SourceError(ValueError):
    """Raised when the synthetic source material is malformed."""


TRANSCRIPT = "transcript"
REVISION = "revision"
EXHIBITION_DRAFT = "exhibition_draft"
NOTE = "note"

RECORD_KINDS = {TRANSCRIPT, REVISION, EXHIBITION_DRAFT, NOTE}


def infer_record_kind(text: str, raw: Mapping[str, Any]) -> str:
    """Infer a record kind without discarding an explicit source value.

    The supplied synthetic data has no ``kind`` field, so the two special
    record types are recognised by their editorial wording.  Explicit metadata
    always wins.
    """

    explicit = raw.get("kind") or raw.get("record_type") or raw.get("type")
    if isinstance(explicit, str):
        value = explicit.strip().lower().replace("-", "_")
        aliases = {
            "transcript": TRANSCRIPT,
            "verbatim": TRANSCRIPT,
            "interview": TRANSCRIPT,
            "revision": REVISION,
            "correction": REVISION,
            "relisten": REVISION,
            "exhibition": EXHIBITION_DRAFT,
            "exhibition_draft": EXHIBITION_DRAFT,
            "draft": EXHIBITION_DRAFT,
            "note": NOTE,
        }
        if value in aliases:
            return aliases[value]
        raise SourceError(f"unknown record kind: {explicit!r}")

    compact = "".join(text.split())
    if compact.startswith(("复听更正", "复听修订", "复听修正", "回听更正")):
        return REVISION
    if "展陈稿" in compact or "展陈方案" in compact:
        return EXHIBITION_DRAFT
    if "更正说明" in compact or "修订说明" in compact:
        return REVISION
    return TRANSCRIPT


def _coerce_int(value: Any, default: int = 1) -> int:
    if value is None or value == "":
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise SourceError(f"version must be an integer, got {value!r}") from exc


def _review_comments(raw: Mapping[str, Any]) -> tuple[ Mapping[str, Any], ...]:
    """Return review comments without normalising away source detail."""

    comments: list[Mapping[str, Any]] = []
    for key in ("review_comments", "review_notes", "comments", "reviews"):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, str):
            comments.append({"text": value})
        elif isinstance(value, Mapping):
            comments.append(dict(value))
        elif isinstance(value, Iterable):
            for item in value:
                if isinstance(item, str):
                    comments.append({"text": item})
                elif isinstance(item, Mapping):
                    comments.append(dict(item))
    return tuple(comments)


@dataclass(frozen=True)
class SourceRecord:
    """One immutable source row.

    ``version`` is the editorial version of the record within an interview.
    A revision note is a new row; it never overwrites the transcript row it
    corrects.
    """

    record_id: str
    interview: str
    version: int
    text: str
    raw: Mapping[str, Any]
    kind: str
    revision_of: str | None = None
    speaker: str | None = None
    status: str | None = None
    review_comments: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    @property
    def is_testimony(self) -> bool:
        return self.kind in {TRANSCRIPT, REVISION}

    @property
    def source_version_label(self) -> str:
        return f"{self.interview} v{self.version} ({self.record_id})"


def _revision_target(raw: Mapping[str, Any]) -> str | None:
    for key in ("revision_of", "supersedes", "corrects", "target_record", "target_record_ids"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
            return value[0].strip()
    value = raw.get("target_record_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def load_source_records(path_or_data: str | bytes | "os.PathLike[str]" | Iterable[Mapping[str, Any]]) -> list[SourceRecord]:
    """Load and validate source records.

    ``path_or_data`` may be a path, JSON bytes, or an already-decoded list.
    Validation is intentionally structural: the tool is a text-indexing aid,
    not a historical adjudicator.
    """

    import os

    if isinstance(path_or_data, (str, bytes, os.PathLike)):
        import json
        from pathlib import Path

        if isinstance(path_or_data, bytes):
            text = path_or_data.decode("utf-8")
        elif isinstance(path_or_data, os.PathLike):
            text = Path(path_or_data).read_text(encoding="utf-8")
        else:
            text = path_or_data
            candidate = Path(text)
            if candidate.exists():
                text = candidate.read_text(encoding="utf-8")
        data = json.loads(text)
    else:
        data = list(path_or_data)

    if isinstance(data, Mapping):
        data = data.get("records")
    if not isinstance(data, list):
        raise SourceError("source material must be a JSON list or an object with a 'records' list")

    records: list[SourceRecord] = []
    seen: set[str] = set()
    for index, item in enumerate(data):
        if not isinstance(item, Mapping):
            raise SourceError(f"record #{index} must be an object")
        record_id = str(item.get("id", "")).strip()
        text = item.get("text")
        if not record_id:
            raise SourceError(f"record #{index} is missing a non-empty 'id'")
        if record_id in seen:
            raise SourceError(f"duplicate record id: {record_id}")
        if not isinstance(text, str) or not text.strip():
            raise SourceError(f"record {record_id} is missing non-empty 'text'")
        seen.add(record_id)
        interview = str(item.get("interview") or item.get("interview_id") or "unassigned").strip()
        version = _coerce_int(item.get("version"), default=1)
        raw = dict(item)
        kind = infer_record_kind(text, raw)
        speaker = raw.get("speaker")
        status = raw.get("status") or raw.get("review_status")
        records.append(
            SourceRecord(
                record_id=record_id,
                interview=interview or "unassigned",
                version=version,
                text=text,
                raw=raw,
                kind=kind,
                revision_of=_revision_target(raw),
                speaker=str(speaker) if speaker is not None else None,
                status=str(status) if status is not None else None,
                review_comments=_review_comments(raw),
            )
        )

    records.sort(key=lambda r: (r.interview, r.version, r.record_id))
    return records


def records_in_interview(records: Iterable[SourceRecord], interview: str) -> list[SourceRecord]:
    return sorted((r for r in records if r.interview == interview), key=lambda r: (r.version, r.record_id))
