import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from oral_history import build_index  # noqa: E402
from oral_history.analysis import detect_evidence, split_clauses, split_sentences  # noqa: E402
from oral_history.index import ArchiveIndex  # noqa: E402
from oral_history.lexicon import load_default_lexicon  # noqa: E402
from oral_history.models import (  # noqa: E402
    EXHIBITION_DRAFT,
    REVISION,
    TRANSCRIPT,
    SourceError,
    infer_record_kind,
    load_source_records,
)


def make_index(rows):
    return ArchiveIndex(load_source_records(rows), load_default_lexicon())


class SourceLoadingTests(unittest.TestCase):
    def test_records_keep_original_text_and_ordering(self):
        records = load_source_records(
            [
                {"id": "b", "interview": "I-1", "version": 2, "text": "第二版"},
                {"id": "a", "interview": "I-1", "version": 1, "text": "第一版"},
            ]
        )
        self.assertEqual([record.record_id for record in records], ["a", "b"])
        self.assertEqual(records[0].text, "第一版")

    def test_revision_is_a_new_record_not_an_overwrite(self):
        records = load_source_records(
            [
                {"id": "t", "interview": "I-1", "version": 1, "text": "我舅舅阿成带我去旧桥。"},
                {"id": "r", "interview": "I-1", "version": 2, "text": "复听更正：是表舅陈成。"},
            ]
        )
        self.assertEqual({record.record_id for record in records}, {"t", "r"})
        self.assertEqual([record.kind for record in records], [TRANSCRIPT, REVISION])

    def test_duplicate_ids_and_empty_text_are_rejected(self):
        with self.assertRaises(SourceError):
            load_source_records(
                [
                    {"id": "x", "text": "一"},
                    {"id": "x", "text": "二"},
                ]
            )
        with self.assertRaises(SourceError):
            load_source_records([{"id": "x", "text": "   "}])

    def test_explicit_kind_wins(self):
        self.assertEqual(
            infer_record_kind("展陈稿", {"kind": "revision"}),
            REVISION,
        )
        self.assertEqual(infer_record_kind("任意文字", {"kind": "exhibition_draft"}), EXHIBITION_DRAFT)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.lexicon = load_default_lexicon()

    def _kind(self, text):
        record = load_source_records([{"id": "x", "text": text}])[0]
        return detect_evidence(text, self.lexicon, record).type

    def test_hearsay_is_not_promoted(self):
        self.assertEqual(self._kind("听老人讲，旧桥很早就拆了。"), "hearsay")

    def test_explicit_denial(self):
        self.assertEqual(self._kind("我没有亲眼见过。"), "not_firsthand")

    def test_firsthand(self):
        self.assertEqual(self._kind("我亲眼见过旧桥。"), "firsthand")

    def test_mixed_markers_stay_mixed(self):
        self.assertEqual(self._kind("我亲眼见过旧桥，也听说它后来拆了。"), "mixed")

    def test_negation_wins_over_firsthand_substring(self):
        self.assertNotEqual(self._kind("那里我没有亲眼见过。"), "firsthand")

    def test_hearsay_scopes_forward_but_firsthand_does_not(self):
        record = load_source_records(
            [{"id": "x", "text": "我亲眼见过旧桥，后来又听说它拆了。"}]
        )[0]
        from oral_history.analysis import analyse_record_text

        clauses = analyse_record_text(record, self.lexicon)
        self.assertEqual([clause.evidence.type for clause in clauses], ["firsthand", "hearsay"])


class AppellationTests(unittest.TestCase):
    def test_appellation_without_name_is_pending(self):
        index = make_index([{"id": "x", "interview": "I", "text": "我舅舅带我去旧桥。"}])
        self.assertEqual(index.relations, [])
        self.assertEqual(len(index.pending), 1)
        self.assertEqual(index.pending[0].reason, "appellation_without_name")

    def test_same_appellation_for_two_different_people_is_not_merged(self):
        index = make_index(
            [
                {"id": "a", "interview": "I", "text": "舅舅阿成带我去旧桥。"},
                {"id": "b", "interview": "I", "text": "舅舅陈成后来搬走了。"},
            ]
        )
        people = {relation.person_id for relation in index.relations}
        self.assertEqual(people, {"person:a-cheng", "person:chen-cheng"})
        self.assertEqual(len(index.relations), 2)

    def test_conjunction_is_not_an_attachment(self):
        index = make_index([{"id": "x", "interview": "I", "text": "我姑姑和阿华去了旧桥。"}])
        self.assertEqual(index.relations, [])
        self.assertTrue(index.pending)

    def test_name_in_another_clause_is_not_an_attachment(self):
        index = make_index(
            [{"id": "x", "interview": "I", "text": "我舅舅去了旧桥，阿成在那里等他。"}]
        )
        self.assertEqual(index.relations, [])

    def test_conjunction_leaves_only_the_directly_attached_name(self):
        index = make_index(
            [{"id": "x", "interview": "I", "text": "表舅陈成和阿成都在。"}]
        )
        # 表舅 attaches to 陈成 only; 阿成 after 和 is a separate mention and
        # must not become a second edge from the same appellation.
        self.assertEqual([relation.person_id for relation in index.relations], ["person:chen-cheng"])

    def test_negated_mention_is_not_an_edge(self):
        index = make_index([{"id": "x", "interview": "I", "text": "不是舅舅阿成。"}])
        self.assertEqual(index.relations, [])


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_old_relation_is_kept_but_superseded(self):
        relations = {relation.person_id: relation for relation in self.index.relations}
        old = relations["person:a-cheng"]
        new = relations["person:chen-cheng"]
        self.assertEqual(old.status, "superseded")
        self.assertEqual(old.superseded_by, new.id)
        self.assertEqual(new.status, "active")

    def test_old_transcript_text_is_unchanged(self):
        record = self.index.by_id["oral-1"]
        self.assertEqual(record.text, "我舅舅阿成带我去旧桥边，那时河还没改道。")

    def test_old_mention_is_marked_superseded(self):
        mention = next(
            mention
            for mention in self.index.mentions
            if mention.record_id == "oral-1" and mention.person_id == "person:a-cheng"
        )
        self.assertEqual(self.index.mention_status(mention), "superseded")

    def test_revision_does_not_rewrite_the_transcript(self):
        report = self.index.interview_report("I-7")
        texts = {record["id"]: record["text"] for record in report["records"]}
        self.assertEqual(texts["oral-1"], self.index.by_id["oral-1"].text)
        self.assertIn("复听更正", texts["oral-2"])

    def test_unmapped_revision_stays_pending(self):
        index = make_index(
            [
                {"id": "t", "interview": "I", "version": 1, "text": "我日常在村里走动。"},
                {"id": "r", "interview": "I", "version": 2, "text": "复听更正：上文有一处说法要改。"},
            ]
        )
        self.assertEqual(index.relations, [])
        self.assertTrue(any(item.kind.startswith("revision") for item in index.pending))

    def test_event_keeps_original_text_and_points_to_correction(self):
        event = next(event for event in self.index.events if event.record_id == "oral-1")
        self.assertEqual(event.status, "revised")
        self.assertIn("阿成", event.text)
        self.assertEqual(event.corrected_person_ids, ("person:chen-cheng",))
        self.assertEqual(event.revision_ids, ("oral-2",))


    def test_explicit_name_correction_supersedes_old_person(self):
        index = make_index(
            [
                {"id": "z1", "interview": "Z", "version": 1, "text": "阿成带我去旧桥。"},
                {"id": "z2", "interview": "Z", "version": 2, "text": "复听更正：阿成应改为陈成。", "revision_of": "z1"},
            ]
        )
        effect = index.revisions[0]
        self.assertEqual(effect.negated_person_ids, ("person:a-cheng",))
        self.assertEqual(effect.corrected_person_ids, ("person:chen-cheng",))
        self.assertEqual(effect.target_basis, "explicit")
        event = index.events[0]
        self.assertEqual(event.status, "revised")
        self.assertIn("person:chen-cheng", event.corrected_person_ids)

    def test_alias_and_dialect_name_resolve_to_one_person(self):
        index = make_index(
            [{"id": "x", "interview": "Z", "text": "三姑是阿华，华仔后来去了旧桥。"}]
        )
        names = {mention.person_id for mention in index.mentions}
        self.assertEqual(names, {"person:a-hua"})


class EventTests(unittest.TestCase):
    def test_event_gets_only_original_time_words(self):
        index = build_index()
        event = next(event for event in index.events if event.record_id == "oral-3")
        self.assertTrue(event.time_phrases)
        self.assertEqual(event.to_dict()["calendar_dates"], [])
        self.assertNotIn("19", json.dumps(event.to_dict()["time_phrases"]))

    def test_denial_clause_is_not_an_event(self):
        index = make_index(
            [{"id": "x", "interview": "I", "text": "听老人讲，旧桥在改道前就拆了；我没有亲眼见过。"}]
        )
        events = index.events_for("I")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].evidence.type, "hearsay")

    def test_hearsay_event_stays_reported(self):
        index = build_index()
        event = next(event for event in index.events if event.record_id == "oral-3")
        self.assertEqual(event.assertion, "reported")
        self.assertFalse(event.evidence.firsthand)

    def test_draft_is_not_an_event_source(self):
        index = build_index()
        self.assertFalse(any(event.record_id == "oral-4" for event in index.events))


class CitationTests(unittest.TestCase):
    def test_draft_claiming_firsthand_from_hearsay_is_flagged(self):
        index = build_index()
        citation = index.citations[0]
        self.assertEqual(citation.claimed_evidence, "firsthand")
        self.assertEqual(citation.evaluation_status, "evidence_mismatch")
        self.assertIn("source_is_hearsay", citation.issues)
        self.assertIn("archive_review_incomplete", citation.issues)

    def test_citation_records_version_and_source_records(self):
        index = build_index()
        citation = index.citations[0]
        self.assertEqual(citation.cited_version, 1)
        self.assertEqual(citation.cited_record_ids, ("oral-3",))

    def test_later_revision_marks_citation_stale_without_deleting_it(self):
        index = make_index(
            [
                {"id": "t1", "interview": "I", "version": 1, "text": "我舅舅阿成带我去旧桥。"},
                {"id": "d", "interview": "D", "version": 1, "text": "展陈稿引用 I，文中作为亲历证言。"},
                {"id": "t2", "interview": "I", "version": 2, "text": "复听更正：是表舅陈成，不是阿成。"},
            ]
        )
        self.assertEqual(len(index.citations), 1)
        citation = index.citations[0]
        self.assertTrue(citation.stale)
        self.assertEqual(citation.affected_by_revision_ids, ("t2",))
        self.assertEqual(citation.draft_record_id, "d")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.index = build_index()

    def test_interview_report_has_all_views(self):
        report = self.index.interview_report("I-7")
        for key in (
            "records",
            "persons",
            "relations",
            "event_timeline",
            "pending",
            "revisions",
            "exhibition_citations",
        ):
            self.assertIn(key, report)

    def test_editor_report_shows_revision_status_and_reasons(self):
        report = self.index.editor_report("I-8")
        entry = report["interviews"][0]
        citation = entry["citations"][0]
        self.assertEqual(citation["evaluation_status"], "evidence_mismatch")
        self.assertTrue(citation["issue_details"])
        self.assertTrue(entry["pending"])

    def test_researcher_report_marks_synthetic(self):
        report = self.index.researcher_report()
        self.assertTrue(report["synthetic"])
        self.assertTrue(all(item["synthetic"] for item in report["interviews"]))

    def test_pending_queue_lists_reasons(self):
        queue = self.index.review_queue()
        self.assertEqual(queue["count"], len(queue["items"]))
        self.assertTrue(queue["items"])
        self.assertTrue(all(item["detail"] for item in queue["items"]))

    def test_search_hits_point_to_source(self):
        hits = self.index.search("阿成")["hits"]
        source_hits = [hit for hit in hits if hit["type"] == "source_text"]
        self.assertEqual({hit["record_id"] for hit in source_hits}, {"oral-1", "oral-2"})
        for hit in source_hits:
            record = self.index.by_id[hit["record_id"]]
            self.assertEqual(
                record.text[hit["span"]["start"]:hit["span"]["end"]],
                hit["span"]["text"],
            )

    def test_index_is_repeatable(self):
        first = json.dumps(build_index().to_dict(), ensure_ascii=False, sort_keys=True)
        second = json.dumps(build_index().to_dict(), ensure_ascii=False, sort_keys=True)
        self.assertEqual(first, second)

    def test_missing_interview_raises(self):
        with self.assertRaises(KeyError):
            self.index.interview_report("no-such-interview")


class SplittingTests(unittest.TestCase):
    def test_offsets_round_trip(self):
        text = "我舅舅阿成带我去旧桥边，那时河还没改道。听老人讲，桥拆了。"
        for sentence in split_sentences(text):
            self.assertEqual(text[sentence.start:sentence.end], sentence.text)
            for clause in split_clauses(sentence):
                self.assertEqual(text[clause.start:clause.end], clause.text)

    def test_synthetic_source_is_marked(self):
        data = json.loads((ROOT / "data" / "oral_history.json").read_text(encoding="utf-8"))
        self.assertIsInstance(data, list)
        report = build_index().researcher_report()
        self.assertTrue(report["synthetic"])


if __name__ == "__main__":
    unittest.main()
