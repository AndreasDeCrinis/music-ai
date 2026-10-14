"""Persistent melodic identity, independent of final instrument assignment."""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .rhythm import Rhythm, seconds_to_quarters


@dataclass(frozen=True)
class VoiceLink:
    voice: str
    predecessor: int | None
    strength: float
    evidence: str


def motif(phrase, rhythm: Rhythm | None):
    if len(phrase.notes) < 6:
        return None
    notes = phrase.notes[:6]
    intervals = tuple(b.pitch - a.pitch for a, b in zip(notes, notes[1:]))
    times = [seconds_to_quarters(n.start, rhythm) if rhythm else n.start * 2 for n in notes]
    iois = tuple(round((b - a) * 2) / 2 for a, b in zip(times, times[1:]))
    return intervals, iois


def link_voices(
    phrases, rhythm: Rhythm | None, known_programs: dict[int, int]
) -> dict[int, VoiceLink]:
    order = sorted(
        range(len(phrases)),
        key=lambda i: (phrases[i].start, -np.mean([n.pitch for n in phrases[i].notes])),
    )
    links = {}
    tips = {}
    histories = {}
    signatures = {i: motif(p, rhythm) for i, p in enumerate(phrases)}
    centers = {i: float(np.median([n.pitch for n in p.notes])) for i, p in enumerate(phrases)}
    programs = {
        i: Counter(
            known_programs[n.program] for n in p.notes if n.program in known_programs
        ).most_common(1)
        for i, p in enumerate(phrases)
    }
    for i in order:
        p = phrases[i]
        candidates = []
        for voice, previous in tips.items():
            prior = phrases[previous]
            if prior.end > p.start + 0.01:
                continue
            gap = (
                (seconds_to_quarters(p.start, rhythm) - seconds_to_quarters(prior.end, rhythm))
                if rhythm
                else (p.start - prior.end) * 2
            )
            matches = [
                old
                for old in histories[voice]
                if signatures[i] is not None
                and signatures[i] == signatures[old]
                and abs(centers[i] - centers[old]) <= 9
            ]
            repeat = bool(matches)
            leap = abs(p.notes[0].pitch - prior.notes[-1].pitch)
            register = abs(centers[i] - centers[previous])
            if repeat:
                leap = min(
                    leap,
                    min(abs(p.notes[0].pitch - phrases[old].notes[0].pitch) for old in matches),
                )
                register = min(register, min(abs(centers[i] - centers[old]) for old in matches))
            if register > 9 or (not repeat and (gap > 24 or leap > 9)):
                continue
            # Isolated one-note predictions provide insufficient identity evidence.
            if len(p.notes) < 2 or len(prior.notes) < 2:
                continue
            mismatch = bool(
                programs[i] and programs[previous] and programs[i][0][0] != programs[previous][0][0]
            )
            cost = (
                leap / 12
                + register / 24
                + min(gap * 0.035, 0.8)
                + 0.4 * mismatch
                - (1.1 if repeat else 0)
            )
            candidates.append((cost, voice, previous, repeat))
        candidates.sort()
        if candidates:
            cost, voice, previous, repeat = candidates[0]
            margin = candidates[1][0] - cost if len(candidates) > 1 else 0.5
            if cost < 1.15 and (repeat or margin >= 0.16):
                strength = min(1.0, 0.55 + margin * 0.6) if repeat else min(0.9, 0.4 + margin * 0.5)
                links[i] = VoiceLink(
                    voice,
                    previous,
                    strength,
                    "repeated melodic motif"
                    if repeat
                    else "register, entry pitch and rest continuity",
                )
                tips[voice] = i
                histories[voice].append(i)
                continue
        voice = f"voice-{len(tips):04d}"
        links[i] = VoiceLink(voice, None, 0.0, "new or ambiguous voice")
        tips[voice] = i
        histories[voice] = [i]
    return links
