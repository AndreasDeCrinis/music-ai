"""Joint, bounded notation fitting with shared attacks and repeated-motif priors."""

from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from .models import Assignment, NoteEvent
from .rhythm import Rhythm, grid_quarters, quantize, seconds_to_quarters


@dataclass(frozen=True)
class FittedNote:
    onset: float
    offset: float
    method: str
    motif_support: int = 0
    unresolved: bool = False


def rational(value: float) -> float:
    return float(Fraction(value).limit_denominator(96))


def triplet_beats(notes: list[NoteEvent], rhythm: Rhythm) -> set[int]:
    """Require both off-beat thirds, rather than enabling tuplets for one late attack."""
    groups = defaultdict(list)
    step = grid_quarters(rhythm.grid)
    for n in notes:
        q = seconds_to_quarters(n.start, rhythm)
        beat = round(q) if abs(q - round(q)) < 0.065 else int(np.floor(q))
        groups[beat].append((n, q))
    supported = set()
    for beat, events in groups.items():
        if len(events) != 3:
            continue
        events.sort(key=lambda x: x[1])
        if all(
            abs(q - (beat + i / 3)) < 0.065
            and abs(seconds_to_quarters(n.end, rhythm) - (beat + (i + 1) / 3)) <= 0.20
            for i, (n, q) in enumerate(events)
        ) and all(abs(q - round(q / step) * step) > 0.045 for _, q in events[1:]):
            supported.add(beat)
    return supported


def snap_candidates(q: float, step: float, thirds: set[int], fine: bool = False) -> list[float]:
    beat = round(q) if abs(q - round(q)) < 0.065 else int(np.floor(q))
    grids = [1 / 3] if beat in thirds else [step]
    if fine and step > 0.125 and beat not in thirds:
        grids.append(max(0.125, step / 2))
    result = set()
    for grid in grids:
        center = round(q / grid)
        for i in range(center - 1, center + 2):
            value = rational(i * grid)
            if value >= 0 and abs(value - q) <= 0.18 + 1e-8:
                result.add(value)
    return sorted(result or {max(0, rational(round(q / step) * step))})


def shared_attacks(
    assignments: list[Assignment],
    rhythm: Rhythm,
    thirds: set[int],
    styles: dict[str, set[int]] | None = None,
) -> dict[str, float]:
    groups = []
    for a in sorted(assignments, key=lambda a: a.note.start):
        q = seconds_to_quarters(a.note.start, rhythm)
        if (
            groups
            and q - groups[-1][0][1] <= 0.065
            and a.part not in {x.part for x, _ in groups[-1]}
        ):
            groups[-1].append((a, q))
        else:
            groups.append([(a, q)])
    anchors = {}
    step = grid_quarters(rhythm.grid)
    for group in groups:
        if len(group) < 2:
            continue
        median = float(np.median([q for _, q in group]))
        onset = min(snap_candidates(median, step, thirds), key=lambda q: abs(q - median))
        if (
            styles is not None
            and abs(onset - round(onset / step) * step) > 1e-7
            and not all(int(np.floor(median)) in styles[a.note.id] for a, _ in group)
        ):
            continue
        # Bounded for every member, including groups straddling a rounding boundary.
        if all(abs(onset - q) <= 0.18 for _, q in group):
            anchors.update({a.note.id: onset for a, _ in group})
    return anchors


def motif_priors(notes: list[NoteEvent], rhythm: Rhythm) -> dict[str, tuple[float, int]]:
    groups = defaultdict(list)
    thirds = triplet_beats(notes, rhythm)
    for i in range(len(notes) - 4):
        window = notes[i : i + 5]
        onsets = np.array([seconds_to_quarters(n.start, rhythm) for n in window])
        iois = np.diff(onsets)
        if np.min(iois) < 0.1 or np.max(iois) > 4:
            continue
        intervals = tuple(b.pitch - a.pitch for a, b in zip(window, window[1:]))
        coarse = tuple(max(0.25, round(float(t) * 2) / 2) for t in iois)
        groups[intervals, coarse].append((i, window))
    priors = {}
    for matches in groups.values():
        independent = []
        for index, window in matches:
            if not independent or index >= independent[-1][0] + 5:
                independent.append((index, window))
        if len(independent) < 2:
            continue
        for position in range(5):
            values = [
                seconds_to_quarters(w[position].end, rhythm)
                - seconds_to_quarters(w[position].start, rhythm)
                for _, w in independent
            ]
            median = float(np.median(values))
            if float(np.median(np.abs(np.array(values) - median))) > 0.12:
                continue
            durations = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            if (
                int(np.floor(seconds_to_quarters(independent[0][1][position].start, rhythm)))
                in thirds
            ):
                durations += [1 / 3, 2 / 3]
            canonical = min(
                durations, key=lambda d: (round(abs(d - median), 8), d not in {0.5, 1.0, 2.0, 4.0})
            )
            target = canonical if abs(canonical - median) <= 0.15 else median
            for _, window in matches:
                n = window[position]
                if (
                    abs(
                        seconds_to_quarters(n.end, rhythm)
                        - seconds_to_quarters(n.start, rhythm)
                        - median
                    )
                    <= 0.3
                ):
                    if n.id not in priors or len(independent) > priors[n.id][1]:
                        priors[n.id] = (target, len(independent))
    return priors


def fit_line(
    notes: list[NoteEvent],
    rhythm: Rhythm,
    anchors: dict[str, float],
    thirds: set[int],
    beam_size: int = 24,
    priors: dict[str, tuple[float, int]] | None = None,
) -> dict[str, FittedNote]:
    """Optimize onset, release and intervening rest jointly, preserving every attack."""
    step = grid_quarters(rhythm.grid)
    priors = motif_priors(notes, rhythm) if priors is None else priors
    raw = [
        (seconds_to_quarters(n.start, rhythm), seconds_to_quarters(n.end, rhythm)) for n in notes
    ]
    # cost, preceding release, fitted events
    beam = [(0.0, 0.0, [])]
    for index, (n, (start, end)) in enumerate(zip(notes, raw, strict=True)):
        neighbor = raw[index + 1][0] if index + 1 < len(raw) else None
        rapid = (neighbor is not None and neighbor - start < step * 0.8) or (
            index > 0 and start - raw[index - 1][0] < step * 0.8
        )
        minimum = min(step, 0.125) if rapid else step
        starts = [anchors[n.id]] if n.id in anchors else snap_candidates(start, step, thirds, rapid)
        expanded = []
        prior_duration, support = priors.get(n.id, (end - start, 0))
        for onset in starts:
            ends = set(snap_candidates(end, step, thirds, rapid))
            # Common binary/dotted durations, and thirds only where attacks support them.
            durations = [minimum, step, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
            if int(np.floor(start)) in thirds:
                durations += [1 / 3, 2 / 3]
            ends.update(rational(onset + d) for d in durations if abs(onset + d - end) <= 0.18)
            close_gap = (
                neighbor is not None and -0.025 <= neighbor - end <= 0.10 and end - start >= 0.3
            )
            if close_gap:
                ends.update(snap_candidates(neighbor, step, thirds, rapid))
            if support and abs(onset + prior_duration - end) <= 0.20 + 1e-8:
                ends.update(snap_candidates(onset + prior_duration, step, thirds, rapid))
            if int(np.floor(onset + 1e-7)) in thirds:
                ends = {rational(onset + 1 / 3)}
            for offset in sorted(ends):
                if offset < onset + minimum - 1e-7 or abs(offset - end) > 0.20 + 1e-8:
                    continue
                timing_cost = ((onset - start) / 0.12) ** 2 + 0.45 * ((offset - end) / 0.16) ** 2
                length = offset - onset
                simple = any(abs(length - d) < 1e-7 for d in durations)
                complexity = 0 if simple else 0.12
                motif_cost = 0.4 * ((length - prior_duration) / 0.18) ** 2 if support else 0
                for cost, previous, events in beam:
                    if onset < previous - 1e-7:
                        continue
                    gap = onset - previous
                    raw_gap = start - raw[index - 1][1] if index else start
                    rest_cost = 0.65 if 0 < gap <= step and 0 <= raw_gap < 0.10 else 0
                    item = FittedNote(
                        onset,
                        offset,
                        "joint phrase fitting" + ("; shared attack" if n.id in anchors else ""),
                        support,
                    )
                    expanded.append(
                        (
                            cost + timing_cost + complexity + motif_cost + rest_cost,
                            offset,
                            events + [item],
                        )
                    )
        if not expanded:
            # Keep this detection without unbounded timing shifts or deleting it.
            onset, offset = quantize(n.start, n.end, rhythm)
            expanded = [
                (
                    cost + 8,
                    max(previous, offset),
                    events + [FittedNote(onset, offset, "bounded fitting fallback", support, True)],
                )
                for cost, previous, events in beam
            ]
        beam = sorted(expanded, key=lambda x: x[0])[:beam_size]
    return {n.id: event for n, event in zip(notes, beam[0][2], strict=True)}


def fit_assignments(assignments: list[Assignment], rhythm: Rhythm) -> dict[str, FittedNote]:
    pitched = [a for a in assignments if a.part and not a.note.drum]
    by_part = defaultdict(list)
    for a in pitched:
        by_part[a.part].append(a.note)
    lines = []
    for notes in by_part.values():
        lanes = []
        for n in sorted(notes, key=lambda n: (n.start, n.pitch, n.id)):
            lane = next((line for line in lanes if line[-1].end <= n.start + 0.01), None)
            if lane is None:
                lanes.append([n])
            else:
                lane.append(n)
        lines.extend(lanes)
    styles = {}
    for line in lines:
        local = triplet_beats(line, rhythm)
        styles.update({n.id: local for n in line})
    thirds = set().union(*styles.values()) if styles else set()
    anchors = shared_attacks(pitched, rhythm, thirds, styles)
    fitted = {}
    for line in lines:
        local_thirds = styles[line[0].id]
        priors = motif_priors(line, rhythm)
        phrases = []
        for n in line:
            if not phrases or n.start - phrases[-1][-1].end > 1.1:
                phrases.append([n])
            else:
                phrases[-1].append(n)
        for phrase in phrases:
            fitted.update(fit_line(phrase, rhythm, anchors, local_thirds, priors=priors))
    return fitted
