"""Lexicon loading and deterministic surface matching.

The lexicon contains only synthetic vocabulary for the rehearsal corpus.  A
person entry may list aliases, but an appellation (舅舅, 表舅, 二姑 ...) is
deliberately *not* mapped to a person: the same appellation can introduce
different relatives and must remain unresolved until the text says who it is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


class LexiconError(ValueError):
    pass


@dataclass(frozen=True)
class PersonEntry:
    person_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class AppellationEntry:
    term: str
    relation: str
    notes: str = ""


@dataclass(frozen=True)
class PlaceEntry:
    place_id: str
    canonical_name: str
    aliases: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class TermMatch:
    start: int
    end: int
    surface: str


@dataclass(frozen=True)
class PersonMatch(TermMatch):
    person_id: str = ""
    canonical_name: str = ""


@dataclass(frozen=True)
class AppellationMatch(TermMatch):
    relation: str = ""


@dataclass(frozen=True)
class PlaceMatch(TermMatch):
    place_id: str = ""
    canonical_name: str = ""


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Iterable):
        return tuple(str(item) for item in value if str(item).strip())
    raise LexiconError(f"expected a string or list of strings, got {type(value).__name__}")


@dataclass(frozen=True)
class Lexicon:
    persons: tuple[PersonEntry, ...]
    appellations: tuple[AppellationEntry, ...]
    places: tuple[PlaceEntry, ...]
    hearsay_markers: tuple[str, ...]
    firsthand_markers: tuple[str, ...]
    event_markers: tuple[str, ...]
    temporal_markers: tuple[str, ...] = (
        "那时",
        "当时",
        "后来",
        "以前",
        "从前",
        "当年",
        "当初",
        "事先",
        "后来",
        "改道前",
        "改道后",
        "拆之前",
        "拆之后",
    )

    @classmethod
    def load(cls, path: str | Path) -> "Lexicon":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, Mapping):
            raise LexiconError("lexicon must be a JSON object")

        persons: list[PersonEntry] = []
        seen_person_ids: set[str] = set()
        for item in data.get("persons", []):
            if not isinstance(item, Mapping):
                raise LexiconError("every person entry must be an object")
            person_id = str(item.get("id") or "").strip()
            canonical = str(item.get("canonical_name") or "").strip()
            if not person_id or not canonical:
                raise LexiconError("person entries need 'id' and 'canonical_name'")
            if person_id in seen_person_ids:
                raise LexiconError(f"duplicate person id: {person_id}")
            seen_person_ids.add(person_id)
            aliases = _as_tuple(item.get("aliases"))
            persons.append(
                PersonEntry(
                    person_id=person_id,
                    canonical_name=canonical,
                    aliases=tuple(dict.fromkeys((canonical, *aliases))),
                    notes=str(item.get("notes") or ""),
                )
            )

        appellations: list[AppellationEntry] = []
        seen_terms: set[str] = set()
        for item in data.get("appellations", []):
            if not isinstance(item, Mapping):
                raise LexiconError("every appellation entry must be an object")
            term = str(item.get("term") or "").strip()
            relation = str(item.get("relation") or term).strip()
            if not term:
                raise LexiconError("appellation entries need a non-empty 'term'")
            if term in seen_terms:
                raise LexiconError(f"duplicate appellation term: {term}")
            seen_terms.add(term)
            appellations.append(
                AppellationEntry(term=term, relation=relation, notes=str(item.get("notes") or ""))
            )

        places: list[PlaceEntry] = []
        seen_place_ids: set[str] = set()
        for item in data.get("places", []):
            if not isinstance(item, Mapping):
                raise LexiconError("every place entry must be an object")
            place_id = str(item.get("id") or "").strip()
            canonical = str(item.get("canonical_name") or "").strip()
            if not place_id or not canonical:
                raise LexiconError("place entries need 'id' and 'canonical_name'")
            if place_id in seen_place_ids:
                raise LexiconError(f"duplicate place id: {place_id}")
            seen_place_ids.add(place_id)
            places.append(
                PlaceEntry(
                    place_id=place_id,
                    canonical_name=canonical,
                    aliases=tuple(dict.fromkeys((canonical, *_as_tuple(item.get("aliases"))))),
                    notes=str(item.get("notes") or ""),
                )
            )

        return cls(
            persons=tuple(persons),
            appellations=tuple(appellations),
            places=tuple(places),
            hearsay_markers=tuple(_as_tuple(data.get("hearsay_markers"))),
            firsthand_markers=tuple(_as_tuple(data.get("firsthand_markers"))),
            event_markers=tuple(_as_tuple(data.get("event_markers"))),
            temporal_markers=tuple(
                _as_tuple(data.get("temporal_markers")) or cls.__dataclass_fields__["temporal_markers"].default
            ),
        )

    # ------------------------------------------------------------------ terms
    @staticmethod
    def _find_all(
        text: str,
        patterns: Iterable[tuple[str, Any]],
    ) -> list[tuple[int, int, str, Any]]:
        """Find non-overlapping matches, preferring longer surfaces.

        Ties are resolved by the caller's pattern order so output is stable.
        The routine works on exact character offsets into the untouched text.
        """

        ordered = sorted(
            ((surface, payload, order) for order, (surface, payload) in enumerate(patterns) if surface),
            key=lambda row: (-len(row[0]), row[2]),
        )
        claimed = [False] * len(text)
        matches: list[tuple[int, int, str, Any]] = []
        for surface, payload, _order in ordered:
            start = 0
            while True:
                index = text.find(surface, start)
                if index < 0:
                    break
                end = index + len(surface)
                if not any(claimed[index:end]):
                    for position in range(index, end):
                        claimed[position] = True
                    matches.append((index, end, surface, payload))
                start = index + 1
        matches.sort(key=lambda row: (row[0], -len(row[2])))
        return matches

    def find_persons(self, text: str) -> list[PersonMatch]:
        patterns: list[tuple[str, PersonEntry]] = []
        for person in self.persons:
            for alias in person.aliases:
                if len(alias) >= 2:  # single-character names are too ambiguous
                    patterns.append((alias, person))
        return [
            PersonMatch(start, end, surface, person.person_id, person.canonical_name)
            for start, end, surface, person in self._find_all(text, patterns)
        ]

    def find_appellations(self, text: str) -> list[AppellationMatch]:
        patterns = [(entry.term, entry) for entry in self.appellations]
        return [
            AppellationMatch(start, end, surface, entry.relation)
            for start, end, surface, entry in self._find_all(text, patterns)
        ]

    def find_places(self, text: str) -> list[PlaceMatch]:
        patterns: list[tuple[str, PlaceEntry]] = []
        for place in self.places:
            for alias in place.aliases:
                if len(alias) >= 2:
                    patterns.append((alias, place))
        return [
            PlaceMatch(start, end, surface, place.place_id, place.canonical_name)
            for start, end, surface, place in self._find_all(text, patterns)
        ]

    def person_by_id(self, person_id: str) -> PersonEntry | None:
        for person in self.persons:
            if person.person_id == person_id:
                return person
        return None

    def place_by_id(self, place_id: str) -> PlaceEntry | None:
        for place in self.places:
            if place.place_id == place_id:
                return place
        return None


DEFAULT_LEXICON_path = Path(__file__).resolve().parents[1] / "data" / "lexicon.json"


def load_default_lexicon() -> Lexicon:
    return Lexicon.load(DEFAULT_LEXICON_path)
