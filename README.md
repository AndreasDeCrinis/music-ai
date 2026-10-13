# Local brass band transcriber

A working local CLI using the pretrained **YourMT3+ YPTF.MoE+Multi (noPS)** model through MT3-Infer. It converts MP3/WAV/FLAC into unquantized MIDI and a provisional, editable 17-part British brass band conductor score. No paid services or audio uploads are used.

The model cannot reliably distinguish cornet desks or reconstruct the original brass orchestration. Part assignments are deterministic suggestions for editing, with source provenance and explicit uncertainty. Empty parts contain rests; the application never adds pitches to fill staves.

## Install

Python 3.11 is tested. From this directory:

```sh
uv sync --python 3.11
uv run brass-transcribe doctor
uv run brass-transcribe download-model
```

If `uv` is absent, this workspace has a local installation at `.tools/uv-env/bin/uv`; substitute that path for `uv`. Install instructions for another machine: https://docs.astral.sh/uv/getting-started/installation/ . All Python dependencies are project-local. FFmpeg is discovered on PATH, or the bundled ARM64 executable from `imageio-ffmpeg` is used. No Homebrew or system-wide change is needed.

The explicit download command fetches approximately 536 MB into `.cache/models` and validates its SHA-256. Ordinary transcription **never downloads weights implicitly**. `--cache PATH` selects another local cache. `uv sync` needs network access for dependencies; inference works offline once dependencies and weights are cached.

## Run

```sh
uv run brass-transcribe input.mp3 --output ./output
uv run brass-transcribe input.mp3 --output ./output-midi --midi-only
uv run brass-transcribe input.mp3 --output ./output-excerpt \
  --start 30 --duration 20 --tempo 120 --time-signature 4/4 --quantize 1/16
uv run brass-transcribe diagnose ./output
```

Use a new output directory for each successful run. `--no-mscz` skips MuseScore conversion. Conversion explicitly uses the MuseScore Basic sound profile; the installed MuseSounds/Cantai integration caused intermittent CLI crashes with the default profile. This changes the exported score profile, not system settings. `--normalize` enables FFmpeg loudness normalization. `--chunk-seconds 30 --overlap 2` bounds model inference memory; overlapping chunks retain raw MIDI independently. `--device cpu` is explicit; `auto` selects CPU on this Mac. MPS is deliberately rejected by this verified adapter.

`--pickup 1` specifies a one-quarter-note anacrusis (short first bar). `--quantize 1/12` or `1/24` uses triplet grids; dotted values and ties are emitted by music21. There is no automatic pickup detection. Beat tracking estimates global tempo and beat positions; meter defaults provisionally to 4/4 because downbeat/meter inference is not implemented. Manual tempo/meter are recommended for brass recordings with rubato.

Excerpt note times remain **absolute relative to the original recording**, including leading rests in the score. MIDI-import mode uses absolute times unchanged:

```sh
uv run brass-transcribe input.wav --from-midi previous.mid --output output-reused --tempo 120
```

This is an explicit debug/reorchestration path, not model inference. No synthetic backend is exposed as a substitute for the pretrained model. Source separation is reserved behind `--separate`, which currently raises an informative error; conventional stems do not isolate brass desks.

## Outputs

- `full_score.musicxml`: all 17 named instrumental parts, transposing written pitches and MusicXML transposition metadata.
- `full_score.mid`: concert-pitch, quantized score MIDI (unquantized model MIDI in `--midi-only` mode).
- `full_score.mscz`: native score when MuseScore CLI conversion succeeds.
- `parts/<instrument>.musicxml`: all 17 individual parts.
- `intermediate/normalized.wav`: mono 16 kHz excerpt, loudness processing only with `--normalize`.
- `intermediate/model_chunks/chunk_*.mid`: untouched per-chunk model MIDI.
- `intermediate/raw_transcription.mid`: stitched, unquantized notes in original recording time.
- `intermediate/model_notes.json`: model notes, source track/program, timing, velocities and IDs.
- `intermediate/assigned_voices.json`: part assignments, confidence and reasons; unassigned notes retained.
- `intermediate/score_note_provenance.json`: source IDs, sounding/written pitch and quantized timing before measure/tie splitting.
- `diagnostics/transcription_report.json`: tempo/beats, meter assumption, runtime/device, original track ranges, part counts, unassigned notes, collisions and low-confidence assignments.
- `diagnostics/piano_roll.png`: original sounding pitches and timing.

Percussion is retained in raw MIDI and diagnostics as unassigned, but is not yet notated. Velocities are preserved from the model; the model may output fixed velocity. No inferred articulations or expressive dynamics are added.

## British notation

The packaged declarative configuration is `src/brass_transcriber/configs/british_brass_band.yaml`; supply an override with `--config PATH`. It includes practical/extended concert ranges, register priorities, polyphony limits and instrument metadata. Ranges are editable heuristics, not hard physical limits for all players.

Written-to-sounding intervals are: soprano cornet +3 semitones; B♭ cornets/flugelhorn −2; E♭ tenor horns −9; B♭ baritone/euphonium/tenor trombones −14; E♭ bass −21; B♭ bass −26; bass trombone 0. Only bass trombone uses bass clef. The score is built explicitly in written pitch; MusicXML metadata carries the inverse relation, avoiding double transposition. Automated tests verify every instrument through MusicXML and MIDI round trips.

Convention references: [Euphonium Studio](https://euphonium.studio/notation/), [OpenStax-derived transposing instruments material](https://opencurriculum.org/5567/transposing-instruments/), and [Stape Silver Band instrumentation](https://www.stapesilverband.co.uk/post/instruments-of-a-brass-band).

## Backend investigation and licenses

Inspected actual source, installation requirements, checkpoint catalogue and license files on 2026-10-13:

- [mimbres/YourMT3](https://github.com/mimbres/YourMT3): public GitHub repository currently contains README/license, not a standalone inference package. Its GitHub license is GPL-3.0. The usable implementation and pretrained checkpoints are in the linked [Hugging Face Space](https://huggingface.co/spaces/mimbres/YourMT3/tree/main).
- [MT3-Infer](https://github.com/openmirlab/mt3-infer), pinned commit `d8cbd287470291dee335f84315c5a9b9ff6e6c8a`: MIT wrapper with Apache-2.0 vendored YourMT3 Space code. Selected backend is exactly the YourMT3+ MoE model, not an alternative model. CPU inference is verified here. Its `utils/framework.py` explicitly rejects MPS; hardware MPS availability alone does not establish model support. CUDA is not tested or exposed by this Mac MVP.
- [mqtik/muse](https://github.com/mqtik/muse), inspected commit `ff37ba5f03c60afa15c5f41e62acc4275ffb3f94`: MIT desktop application; `python/pipeline.py:transcribe_yourmt3` loads the YourMT3 model on CPU, then applies dynamic quantization. Its general GPU/MPS feature claim does not prove MPS support for this model. The narrower MT3-Infer integration avoids importing the piano/Demucs/Tauri stack.

The selected checkpoint comes directly from the author's Space at `amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt`. SHA-256: `ae38e415c79efd5592dcb9b658cdb99ddb11d4c4e1eaa364cab04a052473fc25`. The Space implementation has Apache-2.0 licensing; no separate checkpoint-specific license was found in the inspected package catalogue. Check upstream terms before redistribution; weights are not included in this repository. MT3-Infer also ships a different, unlicensed upstream MT3-PyTorch tree; this application does not select that backend.

Transformers 4.43.4 is pinned to the model's supported compatibility path. `uv.lock` records the tested dependency resolution.

## Validation and limitations

```sh
uv run pytest -q
uv run ruff check src tests
uv run brass-transcribe examples/guitar_excerpt.mp3 --duration 4 \
  --tempo 120 --output output-example
```

The included sample is an excerpt of MT3-Infer's `assets/Chill_Guitar_loop_BANDLAB.wav`, distributed in its MIT repository. It is **not a brass-band benchmark**. `examples/validation.json` records this machine's actual execution evidence. Generated run artifacts are in `output-real`, `output-e2e`, and `output-final` locally and are git-ignored.

Phrase assignment uses beam search over chronological note events. Costs consider model programs, ranges/register, melodic leaps, duration continuity, simultaneous-note capacity and voice crossing. It retains unassigned notes when constraints exclude all plausible parts. Confidence measures heuristic alternative separation, not calibrated reconstruction probability. Scores are prominently labeled provisional.

Highest-priority improvements: evaluate annotated brass-band recordings; calibrate assignment confidence and stronger phrase/harmony constraints; add beat-aligned variable-tempo quantization; improve long-note chunk stitching; add percussion notation and richer MuseScore instrument templates. Fixed-tempo grids cannot faithfully preserve rubato. Sustained-note stitching is heuristic, and quantization can alter durations or create extra notation voices at collapsed attacks. XML validity is not a claim of transcription accuracy.
