import math
from dataclasses import dataclass, field

from .models import Assignment, NoteEvent, PartConfig


@dataclass
class State:
    cost: float = 0
    choices: list[Assignment] = field(default_factory=list)
    history: dict[str, tuple[float, int, float]] = field(default_factory=dict)
    active: dict[str, list[tuple[float, int]]] = field(default_factory=dict)


def assign_voices(
    notes: list[NoteEvent], parts: list[PartConfig], beam_size: int = 24
) -> list[Assignment]:
    """Beam search over chronological phrases, with register/continuity/crossing costs.

    Confidence describes heuristic separation of local alternatives, not probability
    of recovering the original player. All results are provisional.
    """
    beam = [State()]
    for n in sorted(notes, key=lambda n: (n.start, -n.pitch, n.id)):
        if n.drum:
            drum_part = next(
                (p for p in parts if p.percussion and p.extended[0] <= n.pitch <= p.extended[1]),
                None,
            )
            for state in beam:
                state.choices = state.choices + [
                    Assignment(
                        note=n,
                        part=drum_part.id if drum_part else None,
                        confidence=0.7 if drum_part else 0,
                        reason="provisional model percussion flag and GM drum mapping"
                        if drum_part
                        else "percussion retained in raw MIDI; no configured mapping for this pitch",
                    )
                ]
            continue
        expanded = []
        for state in beam:
            candidates = []
            if not n.drum:
                for p in parts:
                    if p.percussion:
                        continue
                    if not p.extended[0] <= n.pitch <= p.extended[1]:
                        continue
                    active = [
                        (end, pitch)
                        for end, pitch in state.active.get(p.id, [])
                        if end > n.start + 0.01
                    ]
                    if len(active) >= p.max_simultaneous:
                        continue
                    cost = abs(n.pitch - p.preferred_pitch) / 12
                    cost += 1.0 if not p.typical[0] <= n.pitch <= p.typical[1] else 0
                    cost += 0 if n.program in p.evidence_programs else 0.8
                    previous = state.history.get(p.id)
                    if previous and n.start - previous[0] < 2:
                        cost += abs(n.pitch - previous[1]) / 18
                        cost += abs(math.log(max(n.end - n.start, 0.05) / previous[2])) * 0.1
                        cost -= 0.6
                    for other in parts:
                        for end, pitch in state.active.get(other.id, []):
                            if (
                                end > n.start
                                and (p.preferred_pitch - other.preferred_pitch) * (n.pitch - pitch)
                                < 0
                            ):
                                cost += 0.2
                    candidates.append((cost, p))
            candidates.sort(key=lambda c: (c[0], c[1].id))
            options = [(c, p) for c, p in candidates[:4]] + [(4.5, None)]
            margin = candidates[1][0] - candidates[0][0] if len(candidates) > 1 else 0
            confidence = min(0.75, 0.3 + max(0, margin) / 3)
            for cost, p in options:
                choice = Assignment(
                    note=n,
                    part=p.id if p else None,
                    confidence=confidence if p else 0,
                    reason="provisional range/register/phrase optimization"
                    if p
                    else "percussion retained in raw MIDI"
                    if n.drum
                    else "range or polyphony constraints",
                )
                history = dict(state.history)
                active_map = dict(state.active)
                if p:
                    history[p.id] = (n.end, n.pitch, n.end - n.start)
                    active_map[p.id] = [
                        (e, v) for e, v in active_map.get(p.id, []) if e > n.start + 0.01
                    ] + [(n.end, n.pitch)]
                expanded.append(
                    State(state.cost + cost, state.choices + [choice], history, active_map)
                )
        beam = sorted(expanded, key=lambda s: s.cost)[:beam_size]
    return beam[0].choices
