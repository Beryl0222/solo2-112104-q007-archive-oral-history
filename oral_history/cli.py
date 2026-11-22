"""Command-line entry point for the oral-history indexer.

Examples
--------
    python3 -m oral_history.cli build --out index.json
    python3 -m oral_history.cli interview I-7
    python3 -m oral_history.cli researcher --interview I-7
    python3 -m oral_history.cli editor --interview I-8
    python3 -m oral_history.cli pending
    python3 -m oral_history.cli search 阿成
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from .index import ArchiveIndex, build_index
from .lexicon import Lexicon, load_default_lexicon
from .models import SourceError, load_source_records

DEFAULT_DATA = Path(__file__).resolve().parents[1] / "data" / "oral_history.json"
DEFAULT_LEXICON = Path(__file__).resolve().parents[1] / "data" / "lexicon.json"


def _emit(payload: Any, out: str | None, indent: int = 2) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=indent, sort_keys=False)
    if out:
        Path(out).write_text(text + "\n", encoding="utf-8")
    else:
        print(text)


def _build(args: argparse.Namespace) -> ArchiveIndex:
    lexicon: Lexicon
    if args.lexicon:
        lexicon = Lexicon.load(args.lexicon)
    else:
        lexicon = load_default_lexicon()
    source = args.data or DEFAULT_DATA
    records = load_source_records(source)
    return ArchiveIndex(records, lexicon)


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", help="source JSON file (default: bundled synthetic material)")
    parser.add_argument("--lexicon", help="lexicon JSON file (default: bundled synthetic vocabulary)")
    parser.add_argument("--out", help="write JSON here instead of stdout")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation (default: 2)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oral_history", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build", help="emit the complete evidence-preserving index")
    _add_common(build)

    researcher = subparsers.add_parser("researcher", help="researcher view: persons, relations, events, pending items")
    _add_common(researcher)
    researcher.add_argument("--interview", help="limit to one interview")

    editor = subparsers.add_parser("editor", help="editor view: cited versions, revision status, review reasons")
    _add_common(editor)
    editor.add_argument("--interview", help="limit to one interview")

    interview = subparsers.add_parser("interview", help="one interview report")
    _add_common(interview)
    interview.add_argument("interview_id", metavar="INTERVIEW")

    person = subparsers.add_parser("person", help="one person across interviews")
    _add_common(person)
    person.add_argument("person_id", metavar="PERSON_ID")

    pending = subparsers.add_parser("pending", help="items that need human confirmation")
    _add_common(pending)
    pending.add_argument("--interview", help="limit to one interview")

    search = subparsers.add_parser("search", help="search original text and derived index")
    _add_common(search)
    search.add_argument("query", metavar="TEXT")
    search.add_argument("--interview", help="limit to one interview")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        index = _build(args)
        if args.command == "build":
            payload = index.to_dict()
        elif args.command == "researcher":
            payload = index.researcher_report(args.interview)
        elif args.command == "editor":
            payload = index.editor_report(args.interview)
        elif args.command == "interview":
            payload = index.interview_report(args.interview_id)
        elif args.command == "person":
            payload = index.person_report(args.person_id)
        elif args.command == "pending":
            payload = index.review_queue(args.interview)
        elif args.command == "search":
            payload = index.search(args.query, args.interview)
        else:  # pragma: no cover - argparse enforces the choices
            parser.error(f"unknown command: {args.command}")
            return 2
    except (SourceError, KeyError, FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _emit(payload, args.out, args.indent)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
