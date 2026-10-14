"""Multi-pass note hypotheses with reversible, conservative selection."""

from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .evidence import PitchEvidence
from .models import NoteEvent, TranscriptionResult

# Additional notes within this onset distance, or overlapping this share of their
# duration, of a kept same-pitch note are treated as the same played note.
DUPLICATE_ONSET = 0.07
DUPLICATE_OVERLAP = 0.3
# Harmonic salience below which a note an octave above a kept note counts as overtone.
OCTAVE_GHOST_SALIENCE = 0.2
# Harmonic salience every pitched note needs, and the stricter bar for primary notes
# that no other pass heard (measured precision of such notes is low).
MIN_SALIENCE = 0.05
SINGLE_PASS_MIN_SALIENCE = 0.2
LISTENER_MIN_SALIENCE = 0.1
# Distinct passes that must agree on a note the primary pass did not hear.
MIN_AGREEMENT = 2


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
    passes: dict[str, TranscriptionResult],
    primary: str,
    evidence: PitchEvidence | None = None,
    trusted: frozenset[str] = frozenset(),
    min_agreement: int | None = None,
) -> tuple[TranscriptionResult, list[NoteEvent], list[dict]]:
    """Select notes; alternate-only notes from `trusted` passes were already filtered."""
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
        # Trusted listeners confirm notes but do not alter YourMT3 timing or labels.
        timed = [(p, n) for p, n in g.observations if p not in trusted] or g.observations
        representative = primary_observation or timed[0][1]
        # Primary model timing is authoritative when available. Median alternate
        # times are used only for additional supported hypotheses.
        start = (
            representative.start
            if primary_observation
            else float(np.median([n.start for _, n in timed]))
        )
        end = (
            representative.end
            if primary_observation
            else float(np.median([n.end for _, n in timed]))
        )
        votes = Counter(n.program for _, n in timed)
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

        def decide(support: int) -> tuple[bool, str]:
            if n.drum:
                keep = primary_observation is not None or support >= 2
                return keep, (
                    "primary drum detection or agreement of distinct passes"
                    if keep
                    else "unsupported alternate-only drum hit"
                )
            if salience is not None and salience < MIN_SALIENCE:
                return False, "no harmonic evidence for this pitch"
            if primary_observation is not None:
                weak = support == 1 and salience is not None and salience < SINGLE_PASS_MIN_SALIENCE
                return not weak, (
                    "single-pass note with weak harmonic evidence"
                    if weak
                    else "retained primary-model note"
                )
            keep = support >= (min_agreement or MIN_AGREEMENT) or (
                salience is not None and salience >= 0.65 and end - start >= 0.12
            )
            return keep, (
                "alternate-pass agreement or strong harmonic evidence"
                if keep
                else "unsupported alternate-only note"
            )

        base_support = len({p for p, _ in g.observations if p not in trusted})
        if base_support:
            keep, reason = decide(support)
        elif salience is not None and salience < LISTENER_MIN_SALIENCE:
            keep, reason = False, "listener note with weak harmonic evidence"
        else:
            keep, reason = True, "separated upper-register listener note"
        # Notes kept only because of a trusted listener are placed after, never
        # instead of, the YourMT3 assignment so they cannot displace other parts.
        listener_dependent = keep and not (base_support and decide(base_support)[0])
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
                listener_dependent=listener_dependent,
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
    # Passes disagree slightly on onsets and note splits, so one played note can form
    # several hypotheses. Additional (non-primary) notes must not duplicate a kept
    # same-pitch note; better supported additions are considered first.
    by_id = {d["id"]: d for d in decisions}
    anchors = {}
    for n in selected:
        if not n.drum and by_id[n.id]["primary_support"]:
            anchors.setdefault(n.pitch, []).append(n)
    additions = sorted(
        (n for n in selected if not n.drum and not by_id[n.id]["primary_support"]),
        key=lambda n: (-by_id[n.id]["pass_support"], n.start - n.end, n.start, n.id),
    )
    duplicates = set()
    for n in additions:
        if any(
            abs(other.start - n.start) < DUPLICATE_ONSET
            or min(other.end, n.end) - max(other.start, n.start)
            > DUPLICATE_OVERLAP * (n.end - n.start)
            for other in anchors.get(n.pitch, [])
        ):
            by_id[n.id].update(
                action="reject",
                reason="duplicates a kept same-pitch note",
                listener_dependent=False,
            )
            duplicates.add(n.id)
        else:
            anchors.setdefault(n.pitch, []).append(n)
    selected = [n for n in selected if n.id not in duplicates]
    # An upper octave over a kept note with little harmonic evidence of its own is
    # usually that note's overtone, not a separate player.
    if evidence is not None:
        by_pitch = {}
        for n in selected:
            if not n.drum:
                by_pitch.setdefault(n.pitch, []).append(n)
        ghosts = set()
        for n in selected:
            if n.drum or by_id[n.id]["audio_salience"] is None:
                continue
            if by_id[n.id]["audio_salience"] < OCTAVE_GHOST_SALIENCE and any(
                min(o.end, n.end) - max(o.start, n.start) > 0.5 * (n.end - n.start)
                for o in by_pitch.get(n.pitch - 12, [])
            ):
                by_id[n.id].update(
                    action="reject",
                    reason="octave overtone of a kept note with weak harmonic evidence",
                    listener_dependent=False,
                )
                ghosts.add(n.id)
        selected = [n for n in selected if n.id not in ghosts]
    result = TranscriptionResult(
        notes=all_notes,
        model=f"YourMT3+ multi-pass hypotheses (primary {primary})",
        device=passes[primary].device,
        runtime=sum(r.runtime for r in passes.values()),
        warnings=[
            "Multi-pass agreement and harmonic salience are heuristics, not calibrated accuracy probabilities.",
            "Additional notes require agreement or strong audio evidence; primary notes are preserved except weak short single-pass detections.",
            "Raw predictions from every pass are retained in intermediate/variants/.",
        ],
    )
    return result, selected, decisions
