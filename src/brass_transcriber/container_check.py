"""Build-time smoke test of actual Linux notation and native export; no model download."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from .models import Assignment, EnsemblePreset, NoteEvent, load_parts
from .notation import build_score, export_score, musescore_export
from .rhythm import Rhythm


def main():
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        parts = load_parts(preset=EnsemblePreset.BRASSARANKA)
        assignments = [
            Assignment(
                note=NoteEvent(
                    id=part.id,
                    pitch=36 if part.percussion else part.typical[0],
                    start=0,
                    end=0.5,
                    drum=part.percussion,
                ),
                part=part.id,
                confidence=1,
                reason="Container export check",
            )
            for part in parts
        ]
        score, _ = build_score(
            assignments, parts, Rhythm(tempo=120, detected_tempo=None, beats=[]), 2
        )
        export_score(score, output)
        warning = musescore_export(output)
        if warning:
            raise RuntimeError(warning)
        with ZipFile(output / "full_score.mscz") as archive:
            native = ET.fromstring(
                archive.read(next(n for n in archive.namelist() if n.endswith(".mscx")))
            )
        native_parts = native.findall(".//Score/Part")
        assert len(native_parts) == len(parts), "MuseScore lost an instrument part"
        ids = [p.findtext("Instrument/instrumentId", "") for p in native_parts]
        assert not any("voice" in i or "choir" in i for i in ids), ids
        print("Container smoke check passed: MusicXML, MIDI and 15-part native MuseScore export")


if __name__ == "__main__":
    main()
