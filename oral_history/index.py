"""Build a reproducible, evidence-preserving index over source records.

Design rules enforced here:

* Source rows are immutable. A correction is a new row that points at prior
  rows; the prior row and its review comments remain visible.
* An appellation is only an edge when a single named person is attached to it
  in the same clause. Otherwise the use goes to the review queue.
* Hearsay is never promoted to firsthand. A relation or event keeps the
  evidence label of the sentence it came from.
* Events do not receive invented calendar dates. Only the time words present in
  the transcript are reported.
* Exhibition drafts are editorial records, not testimony; their citations are
  checked against the cited interview and marked stale when the source changes.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .analysis import (
    ASSERTION_LABELS,
    MAX_ATTACHMENT_GAP,
    RELATION_BEFORE_FILLERS,
    AppellationUse,
    Clause,
    Evidence,
    Mention,
    PlaceUse,
    Span,
    analyse_record_text,
    assertion_type,
    attach_appellations,
    correction_pairs,
)
from .lexicon import Lexicon, load_default_lexicon
from .models import (
    EXHIBITION_DRAFT,
    NOTE,
    REVISION,
    TRANSCRIPT,
    SourceRecord,
    load_source_records,
)

SENTENCE_BREAKS = "。！？!?；;\n"
CLAUSE_BREAKS = "，,：:"
NEGATION_BEFORE = ("而不是", "并非", "不是", "而非", "没有", "未")


def _digest(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:12]


def _clause_bounds(text: str, position: int) -> tuple[int, int]:
    start = 0
    for index in range(0, min(position, len(text))):
        if text[index] in CLAUSE_BREAKS + SENTENCE_BREAKS:
            start = index + 1
    end = len(text)
    for index in range(max(position, 0), len(text)):
        if text[index] in CLAUSE_BREAKS + SENTENCE_BREAKS:
            end = index + 1 if text[index] in SENTENCE_BREAKS else index
            break
    return start, end


def _clause_at(record: SourceRecord, start: int) -> Span:
    clause_start, clause_end = _clause_bounds(record.text, start)
    return Span(clause_start, clause_end, record.text[clause_start:clause_end].strip())


def _is_negated(text: str, start: int) -> bool:
    clause_start, _ = _clause_bounds(text, start)
    prefix = text[clause_start:start]
    window = prefix[-12:]
    return any(cue in window for cue in NEGATION_BEFORE)


@dataclass(frozen=True)
class RelationEdge:
    id: str
    interview: str
    record_id: str
    version: int
    relation: str
    person_id: str
    person_name: str
    person_surface: str
    subject_label: str
    start: int
    end: int
    sentence: Span
    evidence: Evidence
    record_kind: str
    status: str = "active"
    superseded_by: str | None = None
    superseded_reason: str = ""
    revision_ids: tuple[str, ...] = ()
    review_comments: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "interview": self.interview,
            "record_id": self.record_id,
            "version": self.version,
            "relation": self.relation,
            "person_id": self.person_id,
            "person_name": self.person_name,
            "person_surface": self.person_surface,
            "subject_label": self.subject_label,
            "span": {"start": self.start, "end": self.end, "text": self.person_surface},
            "sentence": self.sentence.to_dict(),
            "evidence": self.evidence.to_dict(),
            "record_kind": self.record_kind,
            "status": self.status,
            "superseded_by": self.superseded_by,
            "superseded_reason": self.superseded_reason,
            "revision_ids": list(self.revision_ids),
            "review_comments": [dict(comment) for comment in self.review_comments],
        }


@dataclass(frozen=True)
class Event:
    id: str
    interview: str
    record_id: str
    version: int
    start: int
    end: int
    text: str
    event_type: str
    markers: tuple[str, ...]
    person_ids: tuple[str, ...]
    person_surfaces: tuple[str, ...]
    place_ids: tuple[str, ...]
    place_surfaces: tuple[str, ...]
    time_phrases: tuple[str, ...]
    evidence: Evidence
    assertion: str
    status: str = "active"
    revision_ids: tuple[str, ...] = ()
    corrected_person_ids: tuple[str, ...] = ()
    correction_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "interview": self.interview,
            "record_id": self.record_id,
            "version": self.version,
            "span": {"start": self.start, "end": self.end, "text": self.text},
            "event_type": self.event_type,
            "markers": list(self.markers),
            "person_ids": list(self.person_ids),
            "person_surfaces": list(self.person_surfaces),
            "place_ids": list(self.place_ids),
            "place_surfaces": list(self.place_surfaces),
            "time_phrases": list(self.time_phrases),
            "calendar_dates": [],
            "no_calendar_inference": True,
            "evidence": self.evidence.to_dict(),
            "assertion": self.assertion,
            "assertion_label": ASSERTION_LABELS.get(self.assertion, self.assertion),
            "status": self.status,
            "revision_ids": list(self.revision_ids),
            "corrected_person_ids": list(self.corrected_person_ids),
            "correction_note": self.correction_note,
        }


@dataclass(frozen=True)
class PendingItem:
    id: str
    interview: str
    record_id: str
    version: int
    kind: str
    reason: str
    detail: str
    snippet: str
    evidence: Evidence
    surfaces: tuple[str, ...] = ()
    review_comments: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "interview": self.interview,
            "record_id": self.record_id,
            "version": self.version,
            "kind": self.kind,
            "reason": self.reason,
            "detail": self.detail,
            "snippet": self.snippet,
            "evidence": self.evidence.to_dict(),
            "surfaces": list(self.surfaces),
            "review_comments": [dict(comment) for comment in self.review_comments],
        }


@dataclass(frozen=True)
class RevisionEffect:
    revision_record_id: str
    interview: str
    revision_version: int
    target_record_ids: tuple[str, ...]
    target_inferred: bool
    negated_person_ids: tuple[str, ...]
    introduced_relation_ids: tuple[str, ...]
    superseded_relation_ids: tuple[str, ...]
    superseded_mention_keys: tuple[str, ...]
    affected_draft_ids: tuple[str, ...]
    scope: str
    note: str
    corrected_person_ids: tuple[str, ...] = ()
    target_basis: str = "explicit"
    review_comments: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_record_id": self.revision_record_id,
            "interview": self.interview,
            "revision_version": self.revision_version,
            "target_record_ids": list(self.target_record_ids),
            "target_inferred": self.target_inferred,
            "target_basis": self.target_basis,
            "negated_person_ids": list(self.negated_person_ids),
            "corrected_person_ids": list(self.corrected_person_ids),
            "introduced_relation_ids": list(self.introduced_relation_ids),
            "superseded_relation_ids": list(self.superseded_relation_ids),
            "superseded_mention_keys": list(self.superseded_mention_keys),
            "affected_draft_ids": list(self.affected_draft_ids),
            "scope": self.scope,
            "note": self.note,
            "review_comments": [dict(comment) for comment in self.review_comments],
        }


@dataclass(frozen=True)
class Citation:
    draft_record_id: str
    interview_of_draft: str
    cited_interview: str
    cited_record_ids: tuple[str, ...]
    cited_version: int | None
    claimed_evidence: str
    claimed_evidence_label: str
    source_evidence_types: tuple[str, ...]
    review_status: str
    evaluation_status: str
    issues: tuple[str, ...]
    issue_details: tuple[str, ...]
    stale: bool
    affected_by_revision_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_record_id": self.draft_record_id,
            "interview_of_draft": self.interview_of_draft,
            "cited_interview": self.cited_interview,
            "cited_record_ids": list(self.cited_record_ids),
            "cited_version": self.cited_version,
            "claimed_evidence": self.claimed_evidence,
            "claimed_evidence_label": self.claimed_evidence_label,
            "source_evidence_types": list(self.source_evidence_types),
            "review_status": self.review_status,
            "evaluation_status": self.evaluation_status,
            "issues": list(self.issues),
            "issue_details": list(self.issue_details),
            "stale": self.stale,
            "affected_by_revision_ids": list(self.affected_by_revision_ids),
        }


@dataclass(frozen=True)
class RecordView:
    record: SourceRecord
    revision_status: str
    revision_ids: tuple[str, ...]
    derived_mention_count: int
    derived_relation_count: int
    derived_event_count: int
    pending_count: int
    revision_effect: RevisionEffect | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.record.record_id,
            "interview": self.record.interview,
            "version": self.record.version,
            "kind": self.record.kind,
            "text": self.record.text,
            "source_version": self.record.source_version_label,
            "speaker": self.record.speaker,
            "status": self.record.status,
            "revision_of": self.record.revision_of,
            "revision_status": self.revision_status,
            "revision_ids": list(self.revision_ids),
            "review_comments": [dict(comment) for comment in self.record.review_comments],
            "counts": {
                "mentions": self.derived_mention_count,
                "relations": self.derived_relation_count,
                "events": self.derived_event_count,
                "pending": self.pending_count,
            },
            "revision_effect": self.revision_effect.to_dict() if self.revision_effect else None,
        }


class ArchiveIndex:
    """An in-memory, deterministic index. Build it fresh for repeatable runs."""

    def __init__(self, records: Sequence[SourceRecord], lexicon: Lexicon):
        self.records = sorted(records, key=lambda r: (r.interview, r.version, r.record_id))
        self.lexicon = lexicon
        self.by_id = {record.record_id: record for record in self.records}
        self.mentions: list[Mention] = []
        self.appellation_uses: list[AppellationUse] = []
        self.place_uses: list[PlaceUse] = []
        self.relations: list[RelationEdge] = []
        self.events: list[Event] = []
        self.pending: list[PendingItem] = []
        self.revisions: list[RevisionEffect] = []
        self.citations: list[Citation] = []
        self.record_views: dict[str, RecordView] = {}
        self._revision_relation_map: dict[str, dict[str, tuple[str, ...]]] = {}
        self._build()

    # ------------------------------------------------------------------ build
    def _build(self) -> None:
        mentions_by_record: dict[str, list[Mention]] = {}
        uses_by_record: dict[str, list[AppellationUse]] = {}
        places_by_record: dict[str, list[PlaceUse]] = {}
        evidence_by_record: dict[str, list[Evidence]] = {}
        clauses_by_record: dict[str, list[Clause]] = {}

        for record in self.records:
            clauses = analyse_record_text(record, self.lexicon)
            clauses_by_record[record.record_id] = clauses
            evidence_by_record[record.record_id] = [clause.evidence for clause in clauses]
            mentions: list[Mention] = []
            uses: list[AppellationUse] = []
            places: list[PlaceUse] = []
            for clause in clauses:
                if record.kind in {EXHIBITION_DRAFT, NOTE}:
                    continue
                evidence = clause.evidence
                for match in self.lexicon.find_persons(clause.span.text):
                    mentions.append(
                        Mention(
                            record_id=record.record_id,
                            interview=record.interview,
                            version=record.version,
                            start=clause.span.start + match.start,
                            end=clause.span.start + match.end,
                            surface=match.surface,
                            person_id=match.person_id,
                            canonical_name=match.canonical_name,
                            evidence=evidence,
                            sentence=clause.sentence,
                            clause=clause.span,
                            record_kind=record.kind,
                        )
                    )
                for match in self.lexicon.find_appellations(clause.span.text):
                    uses.append(
                        AppellationUse(
                            record_id=record.record_id,
                            interview=record.interview,
                            version=record.version,
                            start=clause.span.start + match.start,
                            end=clause.span.start + match.end,
                            surface=match.surface,
                            relation=match.relation,
                            evidence=evidence,
                            sentence=clause.sentence,
                            clause=clause.span,
                        )
                    )
                for match in self.lexicon.find_places(clause.span.text):
                    places.append(
                        PlaceUse(
                            record_id=record.record_id,
                            interview=record.interview,
                            version=record.version,
                            start=clause.span.start + match.start,
                            end=clause.span.start + match.end,
                            surface=match.surface,
                            place_id=match.place_id,
                            canonical_name=match.canonical_name,
                            sentence=clause.sentence,
                            clause=clause.span,
                            evidence=evidence,
                        )
                    )
            uses, _ = attach_appellations(uses, mentions, record.text)
            mentions_by_record[record.record_id] = mentions
            uses_by_record[record.record_id] = uses
            places_by_record[record.record_id] = places
            self.mentions.extend(mentions)
            self.appellation_uses.extend(uses)
            self.place_uses.extend(places)

        self._build_relations(mentions_by_record, uses_by_record)
        self._build_citations(evidence_by_record)
        self._build_revisions(mentions_by_record)
        self._build_events(places_by_record, clauses_by_record)
        self._apply_revision_impacts()
        self._build_pending()
        self._build_record_views()

    # -------------------------------------------------------------- relations
    def _build_relations(
        self,
        mentions_by_record: Mapping[str, list[Mention]],
        uses_by_record: Mapping[str, list[AppellationUse]],
    ) -> None:
        for record in self.records:
            uses = uses_by_record.get(record.record_id, [])
            mentions = mentions_by_record.get(record.record_id, [])
            for use in uses:
                if use.attached_person_id is None:
                    continue
                person = self.lexicon.person_by_id(use.attached_person_id)
                if person is None:
                    continue
                # Only positive, explicitly attached uses become edges. A
                # negated mention ("不是阿成") is kept for review/impact, not
                # turned into a relationship.
                attached = next(
                    (
                        mention
                        for mention in mentions
                        if mention.person_id == use.attached_person_id
                        and mention.clause == use.clause
                        and mention.start >= use.end
                        and len(record.text[use.end:mention.start]) <= MAX_ATTACHMENT_GAP
                        and all(ch in RELATION_BEFORE_FILLERS | {" "} for ch in record.text[use.end:mention.start])
                    ),
                    None,
                )
                if attached is not None and _is_negated(record.text, attached.start):
                    continue
                relation_id = f"rel:{record.record_id}:{use.start}:{use.attached_person_id}:{use.relation}:{_digest(use.relation, use.attached_person_id, record.record_id)}"
                self.relations.append(
                    RelationEdge(
                        id=relation_id,
                        interview=record.interview,
                        record_id=record.record_id,
                        version=record.version,
                        relation=use.relation,
                        person_id=person.person_id,
                        person_name=person.canonical_name,
                        person_surface=attached.surface if attached else use.surface,
                        subject_label="受访者（未具名）",
                        start=attached.start if attached else use.end,
                        end=attached.end if attached else use.end,
                        sentence=use.sentence,
                        evidence=use.evidence,
                        record_kind=record.kind,
                        status="active",
                        review_comments=record.review_comments,
                    )
                )

    # --------------------------------------------------------------- revisions
    def _revision_targets(self, revision: SourceRecord) -> tuple[list[SourceRecord], str]:
        """Find the old records a revision refers to.

        Returns ``(targets, basis)`` where basis is explicit / surface_match /
        latest_version / none. Only an explicit ``revision_of`` is treated as
        certain; the other bases are recorded so researchers can see why the
        link was made.
        """
        if revision.revision_of:
            target = self.by_id.get(revision.revision_of)
            if target is not None:
                return [target], "explicit"
            return [], "none"
        candidates = [
            record
            for record in self.records
            if record.interview == revision.interview
            and record.version < revision.version
            and record.kind == TRANSCRIPT
        ]
        if not candidates:
            return [], "none"
        revision_surfaces = {mention.surface for mention in self._mentions_of(revision.record_id)}
        matched = [
            record
            for record in candidates
            if any(m.surface in revision_surfaces for m in self._mentions_of(record.record_id))
        ]
        if matched:
            best_version = max(record.version for record in matched)
            return [record for record in matched if record.version == best_version], "surface_match"
        best_version = max(record.version for record in candidates)
        return [record for record in candidates if record.version == best_version], "latest_version"

    def _mentions_of(self, record_id: str) -> list[Mention]:
        return [mention for mention in self.mentions if mention.record_id == record_id]

    def _build_revisions(self, mentions_by_record: Mapping[str, list[Mention]]) -> None:
        for revision in (r for r in self.records if r.kind == REVISION):
            targets, basis = self._revision_targets(revision)
            revision_mentions = mentions_by_record.get(revision.record_id, [])
            negated = [m for m in revision_mentions if _is_negated(revision.text, m.start)]
            # An explicit "华仔应为阿华" statement is a name replacement even
            # without a negation word. The left-hand name is superseded in the
            # old record and the right-hand name is the correction.
            corrections = correction_pairs(revision, revision_mentions)
            replaced_names = [left for left, _right, _connector in corrections]
            negated_person_ids = tuple(
                dict.fromkeys((*(m.person_id for m in negated), *(m.person_id for m in replaced_names)))
            )
            corrected_mentions = [
                m for m in revision_mentions if m not in negated and m not in replaced_names
            ]
            introduced_relation_ids = tuple(
                relation.id
                for relation in self.relations
                if relation.record_id == revision.record_id
                and relation.person_id not in negated_person_ids
            )

            superseded_mentions: list[Mention] = []
            negated_surfaces = {m.surface for m in negated} | {m.surface for m in replaced_names}
            for target in targets:
                for mention in mentions_by_record.get(target.record_id, []):
                    if mention.person_id in negated_person_ids or mention.surface in negated_surfaces:
                        superseded_mentions.append(mention)
            superseded_keys = tuple(f"{m.record_id}:{m.start}:{m.person_id}" for m in superseded_mentions)
            superseded_relation_ids: list[str] = []
            superseded_surfaces = {mention.surface for mention in superseded_mentions}
            target_ids = {target.record_id for target in targets}
            for relation in self.relations:
                if relation.record_id not in target_ids:
                    continue
                if relation.person_id in negated_person_ids or relation.person_surface in superseded_surfaces:
                    superseded_relation_ids.append(relation.id)
                    continue
                # A correction that re-uses the same person but changes the
                # relationship term (舅舅 -> 表舅) supersedes the old edge;
                # keeping both active would present two current relations.
                for introduced_id in introduced_relation_ids:
                    introduced = next(
                        (r for r in self.relations if r.id == introduced_id), None
                    )
                    if (
                        introduced is not None
                        and introduced.person_id == relation.person_id
                        and introduced.relation != relation.relation
                    ):
                        superseded_relation_ids.append(relation.id)
                        break

            # A revision that cannot be tied to a concrete old mention is kept
            # as a manual-review item; the tool does not guess which index rows
            # to change.
            scope = "person_and_relation" if negated_person_ids and targets and superseded_mentions else "unspecified"
            note = ""
            if not targets:
                note = "复听修订未找到可确认的目标记录，未自动改动任何索引。"
            elif not negated_person_ids:
                note = "复听修订未明确否定某个人名，旧版索引保持原状，需人工确认影响范围。"
            elif not superseded_mentions:
                note = "复听修订否定的人名未出现在可确认的旧版记录中，未自动改动任何索引。"

            affected_drafts = tuple(
                citation.draft_record_id
                for citation in self.citations
                if citation.cited_interview == revision.interview
            )
            self.revisions.append(
                RevisionEffect(
                    revision_record_id=revision.record_id,
                    interview=revision.interview,
                    revision_version=revision.version,
                    target_record_ids=tuple(target.record_id for target in targets),
                    target_inferred=basis != "explicit",
                    target_basis=basis,
                    negated_person_ids=negated_person_ids,
                    introduced_relation_ids=introduced_relation_ids,
                    superseded_relation_ids=tuple(dict.fromkeys(superseded_relation_ids)),
                    superseded_mention_keys=superseded_keys,
                    corrected_person_ids=tuple(dict.fromkeys(m.person_id for m in corrected_mentions)),
                    affected_draft_ids=affected_drafts,
                    scope=scope,
                    note=note,
                    review_comments=revision.review_comments,
                )
            )
            # Store the raw relation mapping on the revision for later impact
            # application, keyed by revision id.
            self._revision_relation_map[revision.record_id] = {
                "superseded": tuple(dict.fromkeys(superseded_relation_ids)),
                "introduced": introduced_relation_ids,
                "targets": tuple(target.record_id for target in targets),
            }

        # Resolve superseded_by after all revision effects are known.
        relation_by_id = {relation.id: relation for relation in self.relations}
        for effect in self.revisions:
            mapping = getattr(self, "_revision_relation_map", {}).get(effect.revision_record_id, {})
            introduced = mapping.get("introduced", ())
            for old_id in mapping.get("superseded", ()):
                old = relation_by_id.get(old_id)
                if old is None or old.status != "active":
                    continue
                replacement = introduced[0] if len(introduced) == 1 else None
                updated = RelationEdge(
                    **{
                        **old.__dict__,
                        "status": "superseded",
                        "superseded_by": replacement,
                        "superseded_reason": effect.note or "复听修订明确否定了此次所指的人名。",
                        "revision_ids": tuple(
                            dict.fromkeys((*old.revision_ids, effect.revision_record_id))
                        ),
                    }
                )
                relation_by_id[old_id] = updated
        self.relations = [relation_by_id[relation.id] for relation in self.relations]

    # ------------------------------------------------------------------ events
    def _build_events(
        self,
        places_by_record: Mapping[str, list[PlaceUse]],
        clauses_by_record: Mapping[str, list[Clause]],
    ) -> None:
        # Event extraction is limited to testimony and revision notes: an
        # editorial draft is not a witness statement and must not enter the
        # researcher's event timeline as if it were one.
        for record in self.records:
            if record.kind in {EXHIBITION_DRAFT, NOTE}:
                continue
            events_for_record: list[Event] = []
            for clause in clauses_by_record.get(record.record_id, []):
                self._maybe_event(
                    record,
                    clause,
                    places_by_record.get(record.record_id, []),
                    events_for_record,
                )
            self.events.extend(events_for_record)

    def _maybe_event(
        self,
        record: SourceRecord,
        clause: Clause,
        places: Sequence[PlaceUse],
        bucket: list[Event],
    ) -> None:
        clause_text = clause.span.text
        markers = [marker for marker in self.lexicon.event_markers if marker in clause_text]
        if not markers:
            return
        # Drop markers that occur only inside a negated phrase such as
        # "没有亲眼见过"; a denial is evidence about knowledge, not an event.
        positive_markers = []
        for marker in markers:
            local = clause_text.find(marker)
            prefix = clause_text[:local][-8:]
            if any(cue in prefix for cue in ("没有", "没", "未", "不是", "并非")):
                continue
            positive_markers.append(marker)
        if not positive_markers:
            return
        clause_start = clause.span.start
        clause_end = clause.span.end
        evidence = clause.evidence
        clause_places = [place for place in places if clause_start <= place.start < clause_end]
        clause_mentions = [
            mention
            for mention in self._mentions_of(record.record_id)
            if clause_start <= mention.start < clause_end and not _is_negated(record.text, mention.start)
        ]
        time_phrases = tuple(
            dict.fromkeys(marker for marker in self.lexicon.temporal_markers if marker and marker in clause_text)
        )
        event_type = positive_markers[0]
        event_id = f"evt:{record.record_id}:{clause_start}:{_digest(record.record_id, str(clause_start), event_type)}"
        bucket.append(
            Event(
                id=event_id,
                interview=record.interview,
                record_id=record.record_id,
                version=record.version,
                start=clause_start,
                end=clause_end,
                text=clause_text.strip().strip(SENTENCE_BREAKS).strip(),
                event_type=event_type,
                markers=tuple(positive_markers),
                person_ids=tuple(dict.fromkeys(mention.person_id for mention in clause_mentions)),
                person_surfaces=tuple(dict.fromkeys(mention.surface for mention in clause_mentions)),
                place_ids=tuple(dict.fromkeys(place.place_id for place in clause_places)),
                place_surfaces=tuple(dict.fromkeys(place.canonical_name for place in clause_places)),
                time_phrases=time_phrases,
                evidence=evidence,
                assertion=assertion_type(evidence, record),
            )
        )

    # ----------------------------------------------------------------- pending
    def _build_pending(self) -> None:
        for use in self.appellation_uses:
            if use.attached_person_id is not None:
                continue
            record = self.by_id[use.record_id]
            if use.resolution.endswith("multiple_names"):
                reason = "multiple_candidate_names"
                detail = "同一称谓后出现多个可能人名，未自动选择其中一个。"
            else:
                reason = "appellation_without_name"
                detail = "只有称谓、没有可确认的人名；不能仅凭称谓建立人物关系。"
            self.pending.append(
                PendingItem(
                    id=f"pending:appellation:{use.record_id}:{use.start}:{_digest(use.record_id, str(use.start), use.surface)}",
                    interview=use.interview,
                    record_id=use.record_id,
                    version=use.version,
                    kind="unresolved_appellation",
                    reason=reason,
                    detail=detail,
                    snippet=use.sentence.text.strip(),
                    evidence=use.evidence,
                    surfaces=(use.surface,),
                    review_comments=record.review_comments,
                )
            )

        for effect in self.revisions:
            record = self.by_id[effect.revision_record_id]
            if not effect.target_record_ids:
                self.pending.append(
                    PendingItem(
                        id=f"pending:revision:{effect.revision_record_id}:target",
                        interview=effect.interview,
                        record_id=effect.revision_record_id,
                        version=effect.revision_version,
                        kind="revision_target_unclear",
                        reason="no_target_record",
                        detail=effect.note,
                        snippet=record.text.strip(),
                        evidence=Evidence("revision_statement"),
                        review_comments=record.review_comments,
                    )
                )
            elif effect.scope == "unspecified":
                self.pending.append(
                    PendingItem(
                        id=f"pending:revision:{effect.revision_record_id}:scope",
                        interview=effect.interview,
                        record_id=effect.revision_record_id,
                        version=effect.revision_version,
                        kind="revision_scope_unclear",
                        reason="no_negated_name",
                        detail=effect.note,
                        snippet=record.text.strip(),
                        evidence=Evidence("revision_statement"),
                        review_comments=record.review_comments,
                    )
                )

        for citation in self.citations:
            if not citation.issues:
                continue
            draft = self.by_id[citation.draft_record_id]
            self.pending.append(
                PendingItem(
                    id=f"pending:citation:{citation.draft_record_id}:{_digest(citation.draft_record_id, citation.cited_interview)}",
                    interview=citation.interview_of_draft,
                    record_id=citation.draft_record_id,
                    version=draft.version,
                    kind="exhibition_citation_review",
                    reason=";".join(citation.issues),
                    detail="；".join(citation.issue_details),
                    snippet=draft.text.strip(),
                    evidence=Evidence("editorial_draft"),
                    surfaces=(citation.cited_interview,),
                    review_comments=draft.review_comments,
                )
            )

    # -------------------------------------------------------------- citations
    def _interview_ids(self) -> list[str]:
        return sorted({record.interview for record in self.records if record.interview})

    def _build_citations(self, evidence_by_record: Mapping[str, list[Evidence]]) -> None:
        interview_ids = self._interview_ids()
        for draft in (r for r in self.records if r.kind == EXHIBITION_DRAFT):
            cited = [interview for interview in interview_ids if interview in draft.text and interview != draft.interview]
            if not cited:
                # A draft may cite its own interview only if an explicit
                # citation phrase is present; avoid treating source labels in
                # the draft's own header as a citation.
                cited = [interview for interview in interview_ids if f"引用 {interview}" in draft.text]
            for cited_interview in dict.fromkeys(cited):
                source_records = [
                    record
                    for record in self.records
                    if record.interview == cited_interview and record.kind in {TRANSCRIPT, REVISION}
                ]
                transcripts = [record for record in source_records if record.kind == TRANSCRIPT]
                cited_version = max((record.version for record in transcripts), default=None)
                evidence_types: list[str] = []
                for record in source_records:
                    for evidence in evidence_by_record.get(record.record_id, []):
                        evidence_types.append(evidence.type)
                evidence_types = list(dict.fromkeys(evidence_types))

                claimed = "unknown"
                if "亲历" in draft.text or "亲眼" in draft.text:
                    claimed = "firsthand"
                elif any(word in draft.text for word in ("转述", "听说", "传闻")):
                    claimed = "reported"

                review_status = "incomplete" if any(
                    phrase in draft.text for phrase in ("尚未完成", "未完成", "待复核", "未经复核")
                ) else "recorded"

                issues: list[str] = []
                issue_details: list[str] = []
                if claimed == "firsthand" and not any(t == "firsthand" for t in evidence_types):
                    issues.append("citation_claims_firsthand_but_source_is_not_firsthand")
                    issue_details.append(
                        "展陈稿把该访谈表述为亲历证言，但被引访谈中没有亲历标记，不能按亲历引证。"
                    )
                if claimed == "firsthand" and "not_firsthand" in evidence_types:
                    issues.append("source_explicitly_denies_firsthand")
                    issue_details.append("被引访谈原话明确表示没有亲眼见过或并非亲历。")
                if claimed == "firsthand" and "hearsay" in evidence_types and "firsthand" not in evidence_types:
                    issues.append("source_is_hearsay")
                    issue_details.append("被引访谈原话是听来的转述，不能作为亲历证言。")
                if review_status == "incomplete":
                    issues.append("archive_review_incomplete")
                    issue_details.append("展陈稿注明档案复核尚未完成。")

                revision_records = [
                    record
                    for record in self.records
                    if record.interview == cited_interview and record.kind == REVISION
                ]
                stale = bool(revision_records)
                if stale:
                    issues.append("source_has_later_revision")
                    issue_details.append("被引访谈存在复听修订，引用版本需要重新核对。")

                if any(issue in issues for issue in ("citation_claims_firsthand_but_source_is_not_firsthand", "source_explicitly_denies_firsthand", "source_is_hearsay")):
                    evaluation = "evidence_mismatch"
                elif stale:
                    evaluation = "stale"
                elif review_status == "incomplete":
                    evaluation = "review_incomplete"
                else:
                    evaluation = "ok"

                self.citations.append(
                    Citation(
                        draft_record_id=draft.record_id,
                        interview_of_draft=draft.interview,
                        cited_interview=cited_interview,
                        cited_record_ids=tuple(record.record_id for record in source_records),
                        cited_version=cited_version,
                        claimed_evidence=claimed,
                        claimed_evidence_label={
                            "firsthand": "亲历证言",
                            "reported": "转述/听闻",
                            "unknown": "未注明",
                        }.get(claimed, claimed),
                        source_evidence_types=tuple(evidence_types),
                        review_status=review_status,
                        evaluation_status=evaluation,
                        issues=tuple(issues),
                        issue_details=tuple(issue_details),
                        stale=stale,
                        affected_by_revision_ids=tuple(record.record_id for record in revision_records),
                    )
                )

    # --------------------------------------------------------- revision impact
    def _apply_revision_impacts(self) -> None:
        for effect in self.revisions:
            # Events in the old record are retained with a correction pointer.
            # Only a revision that actually identifies a changed name narrows
            # the affected evidence; an unspecified revision is left for human
            # review and does not relabel unrelated events.
            updated_events: list[Event] = []
            for event in self.events:
                if event.record_id in effect.target_record_ids and effect.scope == "person_and_relation":
                    updated_events.append(
                        Event(
                            **{
                                **event.__dict__,
                                "status": "revised",
                                "revision_ids": tuple(
                                    dict.fromkeys((*event.revision_ids, effect.revision_record_id))
                                ),
                                "corrected_person_ids": tuple(
                                    dict.fromkeys(
                                        (*event.corrected_person_ids, *effect.corrected_person_ids)
                                    )
                                ),
                                "correction_note": effect.note
                                or "复听修订否定了该片段中的原有人名，原句仍保留。",
                            }
                        )
                    )
                else:
                    updated_events.append(event)
            self.events = updated_events

            # An exhibition draft that cites the revised interview becomes
            # stale even when the correction has nothing to do with the quoted
            # sentence: the editor must re-check the exact version.
            updated_citations: list[Citation] = []
            for citation in self.citations:
                if citation.cited_interview == effect.interview:
                    updated_citations.append(
                        Citation(
                            **{
                                **citation.__dict__,
                                "stale": True,
                                "affected_by_revision_ids": tuple(
                                    dict.fromkeys(
                                        (*citation.affected_by_revision_ids, effect.revision_record_id)
                                    )
                                ),
                                "evaluation_status": "stale",
                                "issues": tuple(
                                    dict.fromkeys((*citation.issues, "source_has_later_revision"))
                                ),
                                "issue_details": tuple(
                                    dict.fromkeys(
                                        (
                                            *citation.issue_details,
                                            "复听修订已产生，展陈稿引用的版本需要重新核对。",
                                        )
                                    )
                                ),
                            }
                        )
                    )
                else:
                    updated_citations.append(citation)
            self.citations = updated_citations

    # ------------------------------------------------------------- record views
    def _build_record_views(self) -> None:
        effect_by_revision = {effect.revision_record_id: effect for effect in self.revisions}
        target_revision_ids: dict[str, list[str]] = {}
        for effect in self.revisions:
            for target_id in effect.target_record_ids:
                target_revision_ids.setdefault(target_id, []).append(effect.revision_record_id)
            for draft_id in effect.affected_draft_ids:
                target_revision_ids.setdefault(draft_id, []).append(effect.revision_record_id)

        for record in self.records:
            revision_ids = tuple(dict.fromkeys(target_revision_ids.get(record.record_id, ())))
            if record.record_id in effect_by_revision:
                revision_status = "correction_record"
            elif revision_ids:
                revision_status = "revised"
            elif record.kind == EXHIBITION_DRAFT:
                revision_status = "draft"
            elif record.kind == NOTE:
                revision_status = "note"
            else:
                revision_status = "current"
            self.record_views[record.record_id] = RecordView(
                record=record,
                revision_status=revision_status,
                revision_ids=revision_ids,
                derived_mention_count=sum(1 for m in self.mentions if m.record_id == record.record_id),
                derived_relation_count=sum(1 for r in self.relations if r.record_id == record.record_id),
                derived_event_count=sum(1 for e in self.events if e.record_id == record.record_id),
                pending_count=sum(1 for p in self.pending if p.record_id == record.record_id),
                revision_effect=effect_by_revision.get(record.record_id),
            )

    # ------------------------------------------------------------------ status
    def mention_status(self, mention: Mention) -> str:
        for effect in self.revisions:
            negated_surfaces = {
                m.surface
                for m in self._mentions_of(effect.revision_record_id)
                if _is_negated(self.by_id[effect.revision_record_id].text, m.start)
            }
            if mention.record_id in effect.target_record_ids and (
                f"{mention.record_id}:{mention.start}:{mention.person_id}" in set(effect.superseded_mention_keys)
                or mention.surface in negated_surfaces
            ):
                return "superseded"
            if mention.record_id == effect.revision_record_id and mention.surface in negated_surfaces:
                return "negated"
        if mention.record_kind == REVISION:
            return "correction"
        return "active"

    def mention_dict(self, mention: Mention) -> dict[str, Any]:
        data = mention.to_dict()
        data["status"] = self.mention_status(mention)
        data["assertion"] = assertion_type(mention.evidence, self.by_id[mention.record_id])
        data["assertion_label"] = ASSERTION_LABELS.get(data["assertion"], data["assertion"])
        return data

    # -------------------------------------------------------------------- API
    @property
    def interviews(self) -> list[str]:
        return sorted({record.interview for record in self.records})

    def records_for(self, interview: str) -> list[SourceRecord]:
        return [record for record in self.records if record.interview == interview]

    def active_relations(self, interview: str | None = None) -> list[RelationEdge]:
        return [
            relation
            for relation in self.relations
            if relation.status == "active" and (interview is None or relation.interview == interview)
        ]

    def stale_relations(self, interview: str | None = None) -> list[RelationEdge]:
        return [
            relation
            for relation in self.relations
            if relation.status != "active" and (interview is None or relation.interview == interview)
        ]

    def events_for(self, interview: str | None = None) -> list[Event]:
        events = [event for event in self.events if interview is None or event.interview == interview]
        return sorted(events, key=lambda e: (e.interview, e.version, e.start, e.record_id))

    def pending_for(self, interview: str | None = None) -> list[PendingItem]:
        return [item for item in self.pending if interview is None or item.interview == interview]

    def citations_for(self, interview: str | None = None) -> list[Citation]:
        return [
            citation
            for citation in self.citations
            if interview is None or citation.interview_of_draft == interview
        ]

    def persons_for(self, interview: str) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        for person in self.lexicon.persons:
            mentions = [
                mention
                for mention in self.mentions
                if mention.interview == interview and mention.person_id == person.person_id
            ]
            relations = [
                relation
                for relation in self.relations
                if relation.interview == interview and relation.person_id == person.person_id
            ]
            if not mentions and not relations:
                continue
            result.append(
                {
                    "person_id": person.person_id,
                    "canonical_name": person.canonical_name,
                    "aliases": list(person.aliases),
                    "notes": person.notes,
                    "mentions": [self.mention_dict(mention) for mention in mentions],
                    "relations": [relation.to_dict() for relation in relations],
                    "evidence_types": sorted(
                        {mention.evidence.type for mention in mentions}
                        | {relation.evidence.type for relation in relations}
                    ),
                }
            )
        return result

    def person_report(self, person_id: str) -> dict[str, Any]:
        person = self.lexicon.person_by_id(person_id)
        if person is None:
            raise KeyError(f"unknown person: {person_id}")
        mentions = [mention for mention in self.mentions if mention.person_id == person_id]
        relations = [relation for relation in self.relations if relation.person_id == person_id]
        event_ids = {
            event.id
            for event in self.events
            if person_id in event.person_ids or person_id in event.corrected_person_ids
        }
        return {
            "person_id": person.person_id,
            "canonical_name": person.canonical_name,
            "aliases": list(person.aliases),
            "notes": person.notes,
            "mentions": [self.mention_dict(mention) for mention in mentions],
            "relations": [relation.to_dict() for relation in relations],
            "events": [event.to_dict() for event in self.events if event.id in event_ids],
            "interviews": sorted({mention.interview for mention in mentions}),
            "pending": [
                item.to_dict()
                for item in self.pending
                if person.canonical_name in item.snippet or any(alias in item.snippet for alias in person.aliases)
            ],
        }

    def interview_report(self, interview: str) -> dict[str, Any]:
        records = self.records_for(interview)
        if not records:
            raise KeyError(f"unknown interview: {interview}")
        views = [self.record_views[record.record_id].to_dict() for record in records]
        return {
            "interview": interview,
            "synthetic": True,
            "notice": "全部资料为合成访谈文本；本索引不生成史实结论，仅呈现原话、异文与待核项。",
            "records": views,
            "persons": self.persons_for(interview),
            "relations": {
                "active": [relation.to_dict() for relation in self.active_relations(interview)],
                "superseded": [relation.to_dict() for relation in self.stale_relations(interview)],
            },
            "event_timeline": [event.to_dict() for event in self.events_for(interview)],
            "pending": [item.to_dict() for item in self.pending_for(interview)],
            "revisions": [
                effect.to_dict() for effect in self.revisions if effect.interview == interview
            ],
            "exhibition_citations": [
                citation.to_dict() for citation in self.citations_for(interview)
            ],
            "appellation_uses": [
                use.to_dict() for use in self.appellation_uses if use.interview == interview
            ],
            "counts": {
                "records": len(records),
                "persons": len(self.persons_for(interview)),
                "active_relations": len(self.active_relations(interview)),
                "superseded_relations": len(self.stale_relations(interview)),
                "events": len(self.events_for(interview)),
                "pending": len(self.pending_for(interview)),
            },
        }

    def researcher_report(self, interview: str | None = None) -> dict[str, Any]:
        interviews = [interview] if interview else self.interviews
        unknown = [name for name in interviews if name not in self.interviews]
        if unknown:
            raise KeyError(f"unknown interview: {unknown[0]}")
        return {
            "synthetic": True,
            "notice": "合成口述史演练索引；不做史实判断。",
            "interviews": [self.interview_report(name) for name in interviews],
        }

    def editor_report(self, interview: str | None = None) -> dict[str, Any]:
        interviews = [interview] if interview else self.interviews
        payload: list[dict[str, Any]] = []
        for name in interviews:
            if name not in self.interviews:
                raise KeyError(f"unknown interview: {name}")
            payload.append(
                {
                    "interview": name,
                    "records": [self.record_views[record.record_id].to_dict() for record in self.records_for(name)],
                    "citations": [citation.to_dict() for citation in self.citations_for(name)],
                    "revisions": [effect.to_dict() for effect in self.revisions if effect.interview == name],
                    "pending": [item.to_dict() for item in self.pending_for(name)],
                }
            )
        return {
            "synthetic": True,
            "notice": "编辑视图只显示引用版本、校订状态和待核原因，不将待核内容自动补成完整关系。",
            "interviews": payload,
        }

    def review_queue(self, interview: str | None = None) -> dict[str, Any]:
        items = self.pending_for(interview)
        return {
            "synthetic": True,
            "interview": interview,
            "count": len(items),
            "items": [item.to_dict() for item in items],
        }

    def search(self, query: str, interview: str | None = None) -> dict[str, Any]:
        """Search original text and derived index, preserving every hit's source."""
        needle = query.strip()
        if not needle:
            return {"query": query, "hits": [], "count": 0}
        hits: list[dict[str, Any]] = []
        for record in self.records:
            if interview and record.interview != interview:
                continue
            start = 0
            while True:
                index = record.text.find(needle, start)
                if index < 0:
                    break
                hits.append(
                    {
                        "type": "source_text",
                        "interview": record.interview,
                        "record_id": record.record_id,
                        "version": record.version,
                        "record_kind": record.kind,
                        "span": {"start": index, "end": index + len(needle), "text": needle},
                        "snippet": record.text[max(0, index - 24): index + len(needle) + 24],
                        "record_status": self.record_views[record.record_id].revision_status,
                    }
                )
                start = index + len(needle)
        for relation in self.relations:
            if interview and relation.interview != interview:
                continue
            if needle in relation.person_name or needle in relation.relation or needle in relation.person_surface:
                hits.append({"type": "relation", "interview": relation.interview, **relation.to_dict()})
        for event in self.events:
            if interview and event.interview != interview:
                continue
            if needle in event.text or needle in event.event_type:
                hits.append({"type": "event", "interview": event.interview, **event.to_dict()})
        for item in self.pending:
            if interview and item.interview != interview:
                continue
            if needle in item.snippet or needle in item.detail:
                hits.append({"type": "pending", "interview": item.interview, **item.to_dict()})
        return {"query": query, "count": len(hits), "hits": hits}

    def to_dict(self) -> dict[str, Any]:
        return {
            "synthetic": True,
            "notice": "全部资料为合成访谈文本；索引保留原话与修订，不生成史实结论。",
            "interviews": self.interviews,
            "records": [view.to_dict() for view in self.record_views.values()],
            "mentions": [self.mention_dict(mention) for mention in self.mentions],
            "appellation_uses": [use.to_dict() for use in self.appellation_uses],
            "relations": [relation.to_dict() for relation in self.relations],
            "events": [event.to_dict() for event in self.events_for()],
            "pending": [item.to_dict() for item in self.pending],
            "revisions": [effect.to_dict() for effect in self.revisions],
            "citations": [citation.to_dict() for citation in self.citations],
        }


def build_index(
    source: str | bytes | Iterable[Mapping[str, Any]] | None = None,
    lexicon: Lexicon | None = None,
) -> ArchiveIndex:
    if source is None:
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "data" / "oral_history.json"
    records = [record for record in load_source_records(source)]
    return ArchiveIndex(records, lexicon or load_default_lexicon())
