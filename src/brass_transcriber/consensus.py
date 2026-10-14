"""Multi-pass note hypotheses with reversible, conservative selection."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .evidence import PitchEvidence
from .models import NoteEvent, TranscriptionResult


@dataclass
class Hypothesis:
    observations: list[tuple[str, NoteEvent]] = field(default_factory=list)

    @property
    def anchor(self):
        return self.observations[0][1]


def match_cost(a: NoteEvent, b: NoteEvent) -> float | None:
    if (a.pitch, a.drum) != (b.pitch, b.drum):
        return None
    tolerance = 0.035 if a.drum else 0.07
    if abs(a.start - b.start) > tolerance:
        return None
    if not a.drum:
        overlap = max(0, min(a.end, b.end) - max(a.start, b.start))
        if overlap < 0.5 * min(a.end - a.start, b.end - b.start):
            return None
    return abs(a.start - b.start) + (0 if a.drum else 0.15 * abs(a.end - b.end))


def build_consensus(
    passes: dict[str, TranscriptionResult], primary: str, evidence: PitchEvidence | None = None
) -> tuple[TranscriptionResult, list[NoteEvent], list[dict]]:
    groups = []
    for pass_id in [primary] + [p for p in passes if p != primary]:
        used = set()
        by_pitch = {}
        for index, g in enumerate(groups):
            by_pitch.setdefault((g.anchor.pitch, g.anchor.drum), []).append(index)
        for n in sorted(passes[pass_id].notes, key=lambda n: (n.start, n.pitch, n.id)):
            candidates = []
            for index in by_pitch.get((n.pitch, n.drum), []):
                if index in used:
                    continue
                cost = match_cost(groups[index].anchor, n)
                if cost is not None:
                    candidates.append((cost, index))
            if candidates:
                _, index = min(candidates)
                groups[index].observations.append((pass_id, n))
                used.add(index)
            else:
                groups.append(Hypothesis([(pass_id, n)]))
                index = len(groups) - 1
                by_pitch.setdefault((n.pitch, n.drum), []).append(index)
                used.add(index)
    all_notes = []
    selected = []
    decisions = []
    for index, g in enumerate(groups):
        primary_observation = next((n for p, n in g.observations if p == primary), None)
        representative = primary_observation or g.anchor
        # Primary model timing is authoritative when available. Median alternate
        # times are used only for additional supported hypotheses.
        start = (
            representative.start
            if primary_observation
            else float(np.median([n.start for _, n in g.observations]))
        )
        end = (
            representative.end
            if primary_observation
            else float(np.median([n.end for _, n in g.observations]))
        )
        votes = Counter(n.program for _, n in g.observations)
        program = representative.program if primary_observation else votes.most_common(1)[0][0]
        n = representative.model_copy(
            update=dict(
                id=f"consensus-{index:05d}",
                start=start,
                end=max(start + 0.001, end),
                program=program,
                provenance=[f"{p}:{x.id}" for p, x in g.observations],
            )
        )
        support = len({p for p, _ in g.observations})
        salience = evidence.support(n) if evidence is not None and not n.drum else None
        if n.drum:
            keep = primary_observation is not None or support >= 2
            reason = (
                "primary drum detection or agreement of distinct passes"
                if keep
                else "unsupported alternate-only drum hit"
            )
        elif primary_observation is not None:
            weak = support == 1 and end - start < 0.075 and salience is not None and salience < 0.04
            keep = not weak
            reason = (
                "retained primary-model note"
                if keep
                else "very short, single-pass note with weak harmonic evidence"
            )
        else:
            keep = support >= 2 or (
                salience is not None and salience >= 0.65 and end - start >= 0.12
            )
            reason = (
                "alternate-pass agreement or strong harmonic evidence"
                if keep
                else "unsupported alternate-only note"
            )
        all_notes.append(n)
        if keep:
            selected.append(n)
        decisions.append(
            dict(
                id=n.id,
                action="retain" if keep else "reject",
                reason=reason,
                primary_support=primary_observation is not None,
                pass_support=support,
                audio_salience=salience,
                pitch=n.pitch,
                start=n.start,
                end=n.end,
                percussion=n.drum,
                sources=[
                    dict(
                        pass_id=p,
                        note_id=x.id,
                        pitch=x.pitch,
                        start=x.start,
                        end=x.end,
                        program=x.program,
                    )
                    for p, x in g.observations
                ],
            )
        )
    result = TranscriptionResult(
        notes=all_notes,
        model=f"YourMT3+ multi-pass hypotheses (primary {primary})",
        device="cpu",
        runtime=sum(r.runtime for r in passes.values()),
        warnings=[
            "Multi-pass agreement and harmonic salience are heuristics, not calibrated accuracy probabilities.",
            "Additional notes require agreement or strong audio evidence; primary notes are preserved except weak short single-pass detections.",
            "Raw predictions from every pass are retained in intermediate/variants/.",
        ],
    )
    return result, selected, decisions
