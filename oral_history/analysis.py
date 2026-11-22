"""Deterministic text analysis for oral-history transcripts.

Every derived object points back to the immutable source record and to an exact
character span. Analysis never rewrites a transcript. In particular, an
appellation such as 舅舅 is not an identity: it only becomes a relation when a
unique person name is explicitly attached to it in the same clause.

Evidence is assigned per clause, not per whole sentence, because a single
sentence may mix "我听说" with "我亲眼见过". A hearsay or firsthand marker
scopes forward over the remaining clauses of its sentence; a denial such as
"我没有亲眼见过" stays on its own clause and is never upgraded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lexicon import Lexicon
from .models import EXHIBITION_DRAFT, NOTE, REVISION, SourceRecord

SENTENCE_TERMINATORS = "。！？!?；;\n"
CLAUSE_SEPARATORS = "，,：:"

EVIDENCE_LABELS = {
    "firsthand": "亲历",
    "hearsay": "传闻/转述（非亲历）",
    "not_firsthand": "明确否认亲历",
    "unattributed": "未注明来源",
    "mixed": "同一句内混合来源",
    "revision_statement": "复听修订说明（非事件证言）",
    "editorial_draft": "展陈编辑文本（非访谈证言）",
    "note": "整理备注",
}

ASSERTION_LABELS = {
    "firsthand": "亲历陈述",
    "reported": "传闻/转述",
    "unattributed": "未注明来源",
    "revision": "复听修订",
    "editorial": "展陈文本",
}

RELATION_BEFORE_FILLERS = set("是我为的了")
# Maximum gap between an appellation and a name for the text to count as an
# explicit attachment (e.g. 表舅陈成, 舅舅是阿成). Conjunctions such as 和/与
# are deliberately excluded: 我姑姑和阿华 names two different people.
MAX_ATTACHMENT_GAP = 4

DENIAL_CUES = (
    "没有亲眼",
    "没亲眼",
    "未曾亲眼",
    "并非亲历",
    "不是亲历",
    "没有亲历",
    "未亲历",
    "不在场",
)

CORRECTION_CONNECTORS = ("应改为", "应当改为", "更正为", "改为", "改作", "应为")
# Longest span allowed between a wrong name, its correction connector, and the
# corrected name (e.g. 华仔应为阿华).
MAX_CORRECTION_GAP = 8


@dataclass(frozen=True)
class Evidence:
    type: str
    markers: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return EVIDENCE_LABELS.get(self.type, self.type)

    @property
    def firsthand(self) -> bool:
        return self.type == "firsthand"

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "label": self.label,
            "markers": list(self.markers),
            "firsthand": self.firsthand,
        }


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "text": self.text}


@dataclass(frozen=True)
class Clause:
    span: Span
    sentence: Span
    evidence: Evidence


@dataclass(frozen=True)
class Mention:
    """A person surface form found in one record."""

    record_id: str
    interview: str
    version: int
    start: int
    end: int
    surface: str
    person_id: str
    canonical_name: str
    evidence: Evidence
    sentence: Span
    record_kind: str
    clause: Span = Span(0, 0, "")

    @property
    def span(self) -> Span:
        return Span(self.start, self.end, self.surface)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "interview": self.interview,
            "version": self.version,
            "record_kind": self.record_kind,
            "surface": self.surface,
            "person_id": self.person_id,
            "canonical_name": self.canonical_name,
            "span": self.span.to_dict(),
            "sentence": self.sentence.to_dict(),
            "clause": self.clause.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True)
class AppellationUse:
    record_id: str
    interview: str
    version: int
    start: int
    end: int
    surface: str
    relation: str
    evidence: Evidence
    sentence: Span
    clause: Span = Span(0, 0, "")
    attached_person_id: str | None = None
    attached_surface: str | None = None
    resolution: str = "unresolved"

    @property
    def span(self) -> Span:
        return Span(self.start, self.end, self.surface)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "interview": self.interview,
            "version": self.version,
            "surface": self.surface,
            "relation": self.relation,
            "span": self.span.to_dict(),
            "sentence": self.sentence.to_dict(),
            "clause": self.clause.to_dict(),
            "evidence": self.evidence.to_dict(),
            "attached_person_id": self.attached_person_id,
            "attached_surface": self.attached_surface,
            "resolution": self.resolution,
        }


@dataclass(frozen=True)
class PlaceUse:
    record_id: str
    interview: str
    version: int
    start: int
    end: int
    surface: str
    place_id: str
    canonical_name: str
    sentence: Span
    evidence: Evidence
    clause: Span = Span(0, 0, "")

    @property
    def span(self) -> Span:
        return Span(self.start, self.end, self.surface)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "interview": self.interview,
            "version": self.version,
            "surface": self.surface,
            "place_id": self.place_id,
            "canonical_name": self.canonical_name,
            "span": self.span.to_dict(),
            "sentence": self.sentence.to_dict(),
            "clause": self.clause.to_dict(),
            "evidence": self.evidence.to_dict(),
        }


def correction_pairs(
    record: SourceRecord, mentions: list[Mention]
) -> list[tuple[Mention, Mention, str]]:
    """Find explicit ``wrong 应为 right`` name corrections.

    Only literal correction connectors are accepted. Anything less explicit is
    left to the review queue; the tool never guesses which name is wrong.
    """
    pairs: list[tuple[Mention, Mention, str]] = []
    text = record.text
    ordered = sorted(mentions, key=lambda mention: mention.start)
    for left, right in zip(ordered, ordered[1:]):
        if left.person_id == right.person_id:
            continue
        for connector in CORRECTION_CONNECTORS:
            between = text[left.end:right.start]
            if between.startswith(connector) and len(between) <= MAX_CORRECTION_GAP + len(connector):
                pairs.append((left, right, connector))
                break
    return pairs


def split_sentences(text: str) -> list[Span]:
    """Split on sentence punctuation, preserving absolute offsets."""
    spans: list[Span] = []
    start = 0
    for index, char in enumerate(text):
        if char in SENTENCE_TERMINATORS:
            end = index + 1
            if text[start:end].strip():
                spans.append(Span(start, end, text[start:end]))
            start = end
    if start < len(text) and text[start:].strip():
        spans.append(Span(start, len(text), text[start:]))
    if not spans and text.strip():
        spans.append(Span(0, len(text), text))
    return spans


def split_clauses(sentence: Span) -> list[Span]:
    """Split one sentence on commas/colons, preserving absolute offsets."""
    clauses: list[Span] = []
    start = 0
    text = sentence.text
    for index, char in enumerate(text):
        if char in CLAUSE_SEPARATORS:
            if text[start:index].strip():
                clauses.append(
                    Span(sentence.start + start, sentence.start + index, text[start:index])
                )
            start = index + 1
    if text[start:].strip():
        clauses.append(Span(sentence.start + start, sentence.start + len(text), text[start:]))
    if not clauses and text.strip():
        clauses.append(sentence)
    return clauses


def _record_level_evidence(record: SourceRecord) -> Evidence | None:
    if record.kind == REVISION:
        return Evidence("revision_statement")
    if record.kind == EXHIBITION_DRAFT:
        return Evidence("editorial_draft")
    if record.kind == NOTE:
        return Evidence("note")
    return None


def detect_evidence(text: str, lexicon: Lexicon, record: SourceRecord) -> Evidence:
    """Classify one clause/sentence in isolation."""
    fixed = _record_level_evidence(record)
    if fixed is not None:
        return fixed

    hearsay = tuple(marker for marker in lexicon.hearsay_markers if marker in text)
    firsthand = tuple(marker for marker in lexicon.firsthand_markers if marker in text)

    # "没有亲眼见过" contains "亲眼见过"; negative cues must win.
    if any(cue in text for cue in DENIAL_CUES):
        return Evidence("not_firsthand", tuple(dict.fromkeys((*hearsay, *firsthand))))
    if hearsay and firsthand:
        return Evidence("mixed", tuple(dict.fromkeys((*hearsay, *firsthand))))
    if hearsay:
        return Evidence("hearsay", hearsay)
    if firsthand:
        return Evidence("firsthand", firsthand)
    return Evidence("unattributed", ())


def analyse_record_text(record: SourceRecord, lexicon: Lexicon) -> list[Clause]:
    """Return clauses with evidence scoped within their sentence.

    Hearsay may carry forward over later clauses of its sentence (听老人讲，
    旧桥…拆了), but firsthand never does: 我亲眼见过 X，后来听说 Y must not
    turn Y into a firsthand statement. A denial such as 我没有亲眼见过 is
    local to its clause and is never overridden by an earlier marker.
    """
    fixed = _record_level_evidence(record)
    clauses: list[Clause] = []
    for sentence in split_sentences(record.text):
        inherited_hearsay: list[str] = []
        for clause_span in split_clauses(sentence):
            text = clause_span.text
            if fixed is not None:
                clauses.append(Clause(clause_span, sentence, fixed))
                continue
            own_hearsay = [marker for marker in lexicon.hearsay_markers if marker in text]
            own_firsthand = [marker for marker in lexicon.firsthand_markers if marker in text]
            denial = any(cue in text for cue in DENIAL_CUES)
            if denial:
                evidence = Evidence(
                    "not_firsthand",
                    tuple(dict.fromkeys((*inherited_hearsay, *own_hearsay, *own_firsthand))),
                )
            else:
                hearsay = tuple(dict.fromkeys((*inherited_hearsay, *own_hearsay)))
                firsthand = tuple(own_firsthand)
                if hearsay and firsthand:
                    evidence = Evidence("mixed", tuple(dict.fromkeys((*hearsay, *firsthand))))
                elif hearsay:
                    evidence = Evidence("hearsay", hearsay)
                elif firsthand:
                    evidence = Evidence("firsthand", firsthand)
                else:
                    evidence = Evidence("unattributed", ())
            clauses.append(Clause(clause_span, sentence, evidence))

            # Only hearsay carries forward; it can only downgrade, never
            # promote, evidence in a later clause.
            inherited_hearsay = list(dict.fromkeys((*inherited_hearsay, *own_hearsay)))
    return clauses


def assertion_type(evidence: Evidence, record: SourceRecord) -> str:
    if record.kind == REVISION:
        return "revision"
    if record.kind == EXHIBITION_DRAFT:
        return "editorial"
    if evidence.type == "firsthand":
        return "firsthand"
    if evidence.type in {"hearsay", "not_firsthand", "mixed"}:
        return "reported"
    return "unattributed"


def attach_appellations(
    uses: list[AppellationUse],
    mentions: list[Mention],
    record_text: str,
) -> tuple[list[AppellationUse], list[str]]:
    """Attach an appellation to a named person only when the text is explicit.

    Rule: one named person must occur after the appellation in the same clause,
    with only short linking characters between them (e.g. 表舅陈成). A name in
    a different clause, a conjunction (和/与), or two candidate names leaves the
    use unresolved for manual review.
    """
    resolved: list[AppellationUse] = []
    unresolved_ids: list[str] = []
    by_record: dict[str, list[Mention]] = {}
    for mention in mentions:
        by_record.setdefault(mention.record_id, []).append(mention)

    for use in uses:
        candidates: list[Mention] = []
        for mention in by_record.get(use.record_id, []):
            if mention.clause != use.clause:
                continue
            if mention.start < use.end:
                continue
            gap = record_text[use.end:mention.start]
            if len(gap) > MAX_ATTACHMENT_GAP:
                continue
            if any(ch not in RELATION_BEFORE_FILLERS | {" "} for ch in gap):
                continue
            candidates.append(mention)
        if len(candidates) == 1:
            candidate = candidates[0]
            resolved.append(
                AppellationUse(
                    **{
                        **use.__dict__,
                        "attached_person_id": candidate.person_id,
                        "attached_surface": candidate.surface,
                        "resolution": "explicit",
                    }
                )
            )
        else:
            reason = "multiple_names" if len(candidates) > 1 else "no_name"
            resolved.append(
                AppellationUse(**{**use.__dict__, "resolution": f"unresolved:{reason}"})
            )
            unresolved_ids.append(use.record_id)
    return resolved, unresolved_ids


def contexts_for(record: SourceRecord, position: int, width: int = 24) -> dict[str, Any]:
    """Original-text context around a character position, never normalised."""
    start = max(0, position - width)
    end = min(len(record.text), position + width)
    return {
        "record_id": record.record_id,
        "interview": record.interview,
        "version": record.version,
        "start": start,
        "end": end,
        "text": record.text[start:end],
    }
