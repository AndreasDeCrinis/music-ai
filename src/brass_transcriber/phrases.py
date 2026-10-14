"""Monophonic voice tracking followed by phrase-level constrained part assignment."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import Assignment, NoteEvent, PartConfig


@dataclass
class Phrase:
    id: str
    notes: list[NoteEvent]

    @property
    def start(self):
        return self.notes[0].start

    @property
    def end(self):
        return max(n.end for n in self.notes)


def ambiguous_orphan(phrase: Phrase, options: list[tuple[float, PartConfig]]) -> bool:
    if len(phrase.notes) != 1 or len(options) < 2:
        return False
    n = phrase.notes[0]
    sources = {source.split(":", 1)[0] for source in n.provenance if ":" in source}
    return n.end - n.start < 0.25 and len(sources) < 2 and options[1][0] - options[0][0] < 0.2


def track_phrases(notes: list[NoteEvent]) -> list[Phrase]:
    pitched = sorted((n for n in notes if not n.drum), key=lambda n: (n.start, -n.pitch, n.id))
    chains = []
    index = 0
    while index < len(pitched):
        group = [pitched[index]]
        index += 1
        while index < len(pitched) and pitched[index].start - group[0].start <= 0.025:
            group.append(pitched[index])
            index += 1
        active = [
            i for i, chain in enumerate(chains) if -0.01 <= group[0].start - chain[-1].end <= 1.1
        ]
        costs = np.full((len(group), len(active) + len(group)), 1.8)
        for row, n in enumerate(group):
            for column, chain_index in enumerate(active):
                prior = chains[chain_index][-1]
                overlap = prior.end - n.start
                leap = abs(n.pitch - prior.pitch)
                if overlap > 0.01 or leap > 12:
                    continue
                gap = max(0, n.start - prior.end)
                source_change = 0.18 if n.program != prior.program else 0
                costs[row, column] = leap / 9 + gap * 0.6 + source_change
        rows, cols = linear_sum_assignment(costs)
        for row, column in zip(rows, cols, strict=True):
            if column < len(active) and costs[row, column] < 1.8:
                chains[active[column]].append(group[row])
            else:
                chains.append([group[row]])
    return [Phrase(f"phrase-{i:04d}", chain) for i, chain in enumerate(chains)]


@dataclass
class PhraseState:
    cost: float = 0
    decisions: list[tuple[int, str | None, float]] = field(default_factory=list)
    active: dict[str, list[NoteEvent]] = field(default_factory=dict)
    last: dict[str, NoteEvent] = field(default_factory=dict)


def phrase_cost(phrase: Phrase, part: PartConfig, known_programs: set[int]) -> float | None:
    if part.percussion or any(
        not part.extended[0] <= n.pitch <= part.extended[1] for n in phrase.notes
    ):
        return None
    durations = np.array([n.end - n.start for n in phrase.notes])
    pitches = np.array([n.pitch for n in phrase.notes])
    center = float(np.average(pitches, weights=durations))
    range_penalty = sum(
        n.end - n.start for n in phrase.notes if not part.typical[0] <= n.pitch <= part.typical[1]
    ) / sum(durations)
    votes = Counter(n.program for n in phrase.notes if n.program in known_programs)
    # Ignore unsupported predicted instrument families rather than systematically
    # rewarding an arbitrary brass instrument for "strings" or "guitar" predictions.
    program_penalty = 0.9 * (1 - votes.get(part.program, 0) / sum(votes.values())) if votes else 0
    orphan_penalty = 0.18 if len(phrase.notes) == 1 and durations[0] < 0.12 else 0
    return (
        abs(center - part.preferred_pitch) / 12 + range_penalty + program_penalty + orphan_penalty
    )


def assign_phrases(
    notes: list[NoteEvent], parts: list[PartConfig], beam_size: int = 32
) -> tuple[list[Assignment], dict]:
    phrases = track_phrases(notes)
    order = sorted(
        range(len(phrases)),
        key=lambda i: (phrases[i].start, -np.mean([n.pitch for n in phrases[i].notes])),
    )
    known = {p.program for p in parts if not p.percussion}
    beam = [PhraseState()]
    for phrase_index in order:
        phrase = phrases[phrase_index]
        expanded = []
        for state in beam:
            options = []
            for part in parts:
                cost = phrase_cost(phrase, part, known)
                if cost is None:
                    continue
                occupied = [n for n in state.active.get(part.id, []) if n.end > phrase.start + 0.01]
                if any(
                    sum(
                        previous.start < n.end - 0.01 and previous.end > n.start + 0.01
                        for previous in occupied
                    )
                    >= part.max_simultaneous
                    for n in phrase.notes
                ):
                    continue
                prior = state.last.get(part.id)
                if prior and 0 <= phrase.start - prior.end < 2:
                    cost += abs(phrase.notes[0].pitch - prior.pitch) / 24 - 0.3
                # Crossing is a soft preference: genuine crossings must remain possible.
                for other in parts:
                    for active_note in state.active.get(other.id, []):
                        if (
                            active_note.start <= phrase.start < active_note.end
                            and (part.preferred_pitch - other.preferred_pitch)
                            * (phrase.notes[0].pitch - active_note.pitch)
                            < 0
                        ):
                            cost += 0.035
                options.append((cost, part))
            options.sort(key=lambda x: (x[0], x[1].id))
            if ambiguous_orphan(phrase, options):
                options = []
            for cost, part in options[:6] + [(4.2, None)]:
                confidence = 0
                if part:
                    alternative = min((c for c, p in options if p.id != part.id), default=cost)
                    confidence = min(0.75, max(0.2, 0.35 + (alternative - cost) / 3))
                active = {
                    k: [n for n in v if n.end > phrase.start + 0.01]
                    for k, v in state.active.items()
                }
                last = dict(state.last)
                if part:
                    active[part.id] = active.get(part.id, []) + phrase.notes
                    last[part.id] = phrase.notes[-1]
                # Each phrase contributes one normalized cost so long musical lines
                # cannot be fragmented merely to distribute their notes among desks.
                expanded.append(
                    PhraseState(
                        state.cost + cost,
                        state.decisions + [(phrase_index, part.id if part else None, confidence)],
                        active,
                        last,
                    )
                )
        beam = sorted(expanded, key=lambda s: s.cost)[:beam_size]
    assignments = []
    details = []
    for phrase_index, part, confidence in beam[0].decisions:
        phrase = phrases[phrase_index]
        details.append(
            dict(
                id=phrase.id,
                part=part,
                confidence=confidence,
                source_ids=[n.id for n in phrase.notes],
                start=phrase.start,
                end=phrase.end,
            )
        )
        for n in phrase.notes:
            assignments.append(
                Assignment(
                    note=n,
                    part=part,
                    confidence=confidence,
                    reason=f"provisional whole-phrase optimization: {phrase.id}"
                    if part
                    else f"no sufficiently supported part for {phrase.id}; retained for review",
                )
            )
    for n in notes:
        if n.drum:
            part = next(
                (p for p in parts if p.percussion and p.extended[0] <= n.pitch <= p.extended[1]),
                None,
            )
            assignments.append(
                Assignment(
                    note=n,
                    part=part.id if part else None,
                    confidence=0.7 if part else 0,
                    reason="provisional GM percussion mapping"
                    if part
                    else "unsupported percussion number",
                )
            )
    return assignments, dict(
        phrases=details,
        phrase_count=len(details),
        mean_phrase_notes=float(np.mean([len(p.notes) for p in phrases])) if phrases else 0,
    )
