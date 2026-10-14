# Local brass band transcriber

A working local CLI using the pretrained **YourMT3+ YPTF.MoE+Multi (noPS)** model through MT3-Infer. It converts MP3/WAV/FLAC into unquantized MIDI and provisional, editable conductor scores. Presets include a 17-part British brass band and the 15-part Brassaranka wind ensemble. No paid services or audio uploads are used.

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

## Brassaranka preset

```sh
uv run brass-transcribe "Sunchyme - Brassaranka.mp3" --ensemble brassaranka --output ./sunchyme-brassaranka
# Reuse the existing model output and reassign notes without repeating inference:
uv run brass-transcribe resume ./sunchyme --ensemble brassaranka --output ./sunchyme-brassaranka
```

The preset has E♭ and B♭ clarinets, two B♭ trumpets, four B♭ flugelhorns,
B♭ tenor horn and baritone, three trombones, one tuba, and drum set.
The six trumpet/flugelhorn players listed in the [linked Wikipedia roster](https://de.wikipedia.org/wiki/Brassaranka)
are represented as two trumpets and four flugelhorns according to the band's
[official 2026 technical rider, stage plot on page 5](https://www.brassaranka.at/_files/ugd/00e78a_67909573d10047669d756e0395d977d1.pdf).
Doubling and the exact instrumentation of this particular recording remain uncertain;
this is a sourced default preset, not a claim to recover the original arrangement.
Singing is not inferred or added automatically.

Notation defaults to Austrian conventions: clarinets/trumpets/flugelhorns use
transposing treble clef; B♭ tenor horn and baritone sound a major ninth below
their treble-clef parts; trombones and tuba use concert-pitch bass clef.
The Austrian tenor horn is not the British E♭ alto/tenor horn. The baritone uses
MuseScore's euphonium template as a playback approximation. Change clefs or
transpositions in `src/brass_transcriber/configs/brassaranka.yaml` if your players
use alternative parts. Instrument ranges and register priorities are heuristics.
Each pitched part is monophonic; uncertain excess detections remain unassigned.

Detected General MIDI percussion pitches 35–81 are routed to the drum set,
rendered on percussion clef with simultaneous hits as chords and explicit
per-drum MusicXML/MIDI mappings. Notated hits last one rhythmic grid step;
original model durations remain in the raw transcription. Unsupported drum
numbers remain unassigned and visible in diagnostics. This is provisional kit
notation, not verified drum-performance reconstruction.

British instrumentation remains the default (`--ensemble british-brass-band`).
Use either `--ensemble` or a custom `--config PATH`. An explicit ensemble on
`resume` reassigns notes; metadata-only changes with identical part IDs preserve
assignments. `resume --output NEW_DIR` copies intermediates and leaves the source
run unchanged. Ensemble names, sources, assumptions and instrument settings are
saved in the run metadata and diagnostic report. No new model is downloaded.

If notation export fails after inference, recover the saved run without loading
the model or transcribing the audio again:

```sh
uv run brass-transcribe resume ./output
# Optional notation overrides:
uv run brass-transcribe resume ./output --tempo 120 --time-signature 4/4 --quantize 1/16
# Apply updated instrument metadata while retaining cached notes/assignments:
uv run brass-transcribe resume ./output --config src/brass_transcriber/configs/british_brass_band.yaml
```

New runs save tempo, meter, quantization, excerpt offset and instrumentation in
`intermediate/run_settings.json` before inference. Recovery preserves model
notes, their IDs and existing assignments. Legacy runs without that file recover
tempo from raw MIDI and assume start 0, 4/4 and 1/16, with a diagnostic warning;
use the overrides to restore any different original settings. Resume regenerates
exports in the same output directory. Scores pad the recording's arbitrary
sample-length endpoint with rests to a complete final measure, preventing
unnotatable trailing durations without changing detected notes.

Use a new output directory for each successful run. `--no-mscz` skips MuseScore conversion. Conversion explicitly uses the MuseScore Basic sound profile; the installed MuseSounds/Cantai integration caused intermittent CLI crashes with the default profile. This changes the exported score profile, not system settings. `--normalize` enables FFmpeg loudness normalization. `--chunk-seconds 30 --overlap 2` bounds model inference memory; overlapping chunks retain raw MIDI independently. `--device cpu` is explicit; `auto` selects CPU on this Mac. MPS is deliberately rejected by this verified adapter.

`--pickup 1` specifies a one-quarter-note anacrusis (short first bar). `--quantize 1/12` or `1/24` uses triplet grids; dotted values and ties are emitted by music21. The standard transcription command does not detect pickups automatically. Its beat tracking estimates global tempo and beat positions; meter defaults provisionally to 4/4. The quality pipeline adds neural downbeats and provisional pickup detection, using the configured meter. Manual tempo/meter are recommended when these estimates are uncertain.

Excerpt note times remain **absolute relative to the original recording**, including leading rests in the score. MIDI-import mode uses absolute times unchanged:

```sh
uv run brass-transcribe input.wav --from-midi previous.mid --output output-reused --tempo 120
```

This is an explicit debug/reorchestration path, not model inference. No synthetic backend is exposed as a substitute for the pretrained model. Source separation is reserved behind `--separate`, which currently raises an informative error; conventional stems do not isolate brass desks.

## Precision pipeline

Use `improve` on a saved run to spend more local computation on transcription,
timing and phrase assignment. It creates a separate output and preserves the
baseline. Download the additional weights explicitly once:

```sh
uv run brass-transcribe download-model --variant ps
uv run brass-transcribe download-model --variant beats
uv run brass-transcribe improve ./sunchyme-brassaranka \
  --output ./output-quality-sunchyme --passes 4
```

This uses the alternate YourMT3+ pitch-shift-trained MoE checkpoint, then two
additional passes on audio shifted up/down one semitone with different chunk
boundaries. Detected pitches are shifted back before comparison. The unshifted
PS pass anchors selection: augmentations can miss valid notes, so absence in
those passes does not by itself reject a primary note. Alternate-only additions
require agreement or harmonic-template audio evidence. Every retained/rejected
hypothesis has a decision and its original observations saved for review.
These passes share an architecture and are correlated; agreement is a heuristic,
not independent proof of correctness.

Beat This! estimates beat/downbeat activations, and a madmom HMM decoder imposes
metrical continuity using the configured meter. Notes are quantized against a
shared beat grid. Playback uses rounded, integer-BPM sections, smoothing small
performance timing variations. Sustained changes of at least 6 BPM and about
6% over two or more measures remain visible in the score and extracted parts.
For example, 16 bars at 130 BPM followed by 150 BPM retain a tempo change at
bar 17. Original beat observations remain cached for note placement and review.
Explicit MusicXML forward/backup anchors prevent
MuseScore from dropping tempo changes inside long rests. Ensembles with at most
15 pitched parts get separate MIDI channels per desk, retaining overlapping
unisons independently. The first bar's lead-in and any pickup still need
review. In quality mode, a first downbeat that fits a short whole-beat opening
can suggest a provisional pickup instead of forcing a full first bar and a
false slow opening tempo. Review this inference; explicit nonzero pickup
settings are preserved. `--tempo` deliberately selects constant-tempo timing instead.
Monophonic voice tracking and whole-phrase part assignment reduce isolated
notes switching desks. They cannot identify the original player from a mixed
recording. A short isolated fragment supported by only one pass remains
unassigned when the best two instrument choices have similar costs; its
pitch and observations remain available for review. `--passes 2` skips the
augmented inference passes.

Additional downloads are approximately 724 MB (PS) and 78 MB (beat tracker).
An inference-only copy of PS removes eight training augmentation buffers;
learned parameters are loaded strictly unchanged. Including that prepared copy,
the local weight cache is about 2 GB. All inference remains local on CPU.

Quality runs additionally contain `intermediate/variants/` (raw model MIDI and
normalized notes per pass), `quality_notes.json` (selected hypotheses),
`quality_decisions.json`, `pitch_evidence.npz`, `timing/`, `phrases.json`, and
`diagnostics/quality_report.json`. `diagnostics/opening_comparison.png` shows
the opening assignments and highlights the original second-bar audio window.
Root `model_notes.json` and
`raw_transcription.mid` retain the unfiltered combined hypotheses. Original
baseline files are copied into `intermediate/baseline/` and remain unchanged
in the source run. `resume` preserves quality selection, timing and metadata;
use `resume ./output --output ./output-steady --regularize-tempo` to apply
steady section tempos to an existing cached run without repeating inference.
an explicit new ensemble reassigns selected notes only.

The alternate checkpoint's SHA-256 is
`7427055b51c3c8c86f6a35493cf0741d8db29186052055b59193590a82e6ec01`,
from the author's [YourMT3 Space](https://huggingface.co/spaces/mimbres/YourMT3/tree/main/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2/checkpoints).
[Beat This!](https://github.com/CPJKU/beat_this) code and weights are MIT licensed;
the final0 checkpoint hash is
`8c328b45f59d8dd3dff219253ff6a8d6482be57d0133a29140e2febbf8eb8331`.
The pinned [madmom fork](https://github.com/CPJKU/madmom/tree/27f032e8947204902c675e5e341a3faf5dc86dae)
provides the BSD HMM decoder; its bundled pretrained models are not used.

Structural comparisons in the quality report measure fragmentation and timing,
not actual note accuracy. A reference score or annotated passage is needed to
measure pitch precision and to tune instrument assignment reliably.

## Output files

### Refine rhythm and instrument continuity without inference

```sh
uv run brass-transcribe refine ./output-500-miles --output ./output-500-miles-enhanced
```

This reuses cached predictions and quality selection. The destination must be
separate from the source. Raw notes and rejected candidates stay available.
Previously scored, selected notes are retained. Unassigned candidates remain
available for review. The beam search keeps the complete cached assignment as
a feasible fallback. A required handoff is exempt
from the instrument-switch penalty when the former part is unavailable because
of range or overlap constraints.

Persistent melodic identities link phrases across rests using entry pitch,
register, rhythm and recurring melodic motifs. Returning motifs can reconnect
after an intervening phrase. Similar instrument sounds have a stronger switching
penalty, while genuinely ambiguous identities remain separate. These identities
are provisional, not verified player labels.

Joint phrase fitting chooses attacks, releases and intervening rests together.
Shared ensemble attacks stay aligned. Repeated motifs provide duration priors;
complete, coherent three-note groups within one melodic line enable triplets.
Uncertain or mixed groups use the regular grid. Close repeated attacks can use
32nd notes to avoid a grid collision. Successful fits limit onset movement to
0.18 quarter notes and release movement to 0.20 quarter notes against the
source beat map. An infeasible fit falls back to the configured grid and is
explicitly recorded instead of deleting the detection. Staccato gaps are
retained; short rounding gaps near legato releases receive a penalty.

`diagnostics/refinement_report.json` compares linked-phrase part changes,
extra notation voices and small gaps near legato boundaries. `phrases.json`
records identity links and evidence; score provenance records the fitting
method, repeated-motif support and fallbacks. These measure structural changes,
not reference-score accuracy. `resume` preserves refinement settings, including
when changing tempo. New `improve` runs use these stages automatically.

- `full_score.musicxml`: the selected ensemble's named instrumental parts, written pitches and MusicXML transposition metadata.
- `full_score.mid`: concert-pitch, quantized score MIDI (unquantized model MIDI in `--midi-only` mode).
- `full_score.mscz`: native score when MuseScore CLI conversion succeeds.
- `parts/<instrument>.musicxml`: all individual parts for the selected ensemble.
- `intermediate/normalized.wav`: mono 16 kHz excerpt, loudness processing only with `--normalize`.
- `intermediate/model_chunks/chunk_*.mid`: untouched per-chunk model MIDI.
- `intermediate/raw_transcription.mid`: stitched, unquantized notes in original recording time.
- `intermediate/model_notes.json`: model notes, source track/program, timing, velocities and IDs.
- `intermediate/assigned_voices.json`: part assignments, confidence and reasons; unassigned notes retained.
- `intermediate/score_note_provenance.json`: source IDs, sounding/written pitch and quantized timing before measure/tie splitting.
- `diagnostics/transcription_report.json`: tempo/beats, meter assumption, runtime/device, original track ranges, part counts, unassigned notes, collisions and low-confidence assignments.
- `diagnostics/piano_roll.png`: original sounding pitches and timing.

Percussion is retained in raw MIDI and diagnostics. It is notated when the configuration includes a percussion part, as in Brassaranka; it remains unassigned for the default British pitched-only configuration. Velocities are preserved from the model; the model may output fixed velocity. No inferred articulations or expressive dynamics are added.

## British notation

The packaged declarative configuration is `src/brass_transcriber/configs/british_brass_band.yaml`; supply an override with `--config PATH`. It includes practical/extended concert ranges, register priorities, polyphony limits and instrument metadata. `name` is the displayed staff/desk label; `instrument_name` is a canonical instrument name for MusicXML/MuseScore template recognition. The E♭/B♭ Bass staves explicitly identify tubas, preventing MuseScore from selecting a vocal bass or men's choir sound. Native export tests inspect the actual MuseScore instrument IDs for all 17 parts. Ranges are editable heuristics, not hard physical limits for all players.

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

Further improvements need annotated brass-band recordings, calibrated assignment confidence, stronger harmony constraints and better sustained-note chunk stitching. The standard transcription path uses a fixed-tempo grid; `improve` adds beat-aligned variable tempo and phrase tracking. Sustained-note stitching is heuristic, and quantization can alter durations or create extra notation voices at collapsed attacks. XML validity is not a claim of transcription accuracy.

## Docker web application

Brass Studio serves an MP3 upload page, persistent job history, processing logs,
and downloads of the native MuseScore score (`.mscz`), MusicXML, MIDI and a ZIP
containing the score and all individual MusicXML parts. Brassaranka is the web
default; the British brass band preset is also selectable. Detailed mode uses
four inference passes. Balanced mode uses two; both use neural beat tracking,
shared integer tempo sections, persistent phrase assignment and joint rhythm
fitting. Existing CLI workflows remain available.

### Start with Docker Compose

Install/start your Docker engine, then run from this directory:

```sh
docker compose build
# Explicit one-time download with SHA-256 verification (~1.34 GB downloaded,
# ~1.9 GB of cache including the prepared inference checkpoint):
docker compose --profile setup run --rm models
docker compose up -d app
```

Open **http://localhost:8000**. Once setup finishes, transcription runs offline.
The image includes Python, CPU PyTorch, FFmpeg, the models' inference code,
beat tracking and a headless MuseScore exporter. Model weights are kept in a
persistent `models` volume; uploaded audio, job records and results are in the
`jobs` volume. Rebuilding the app does not erase either volume. You can delete
completed/failed recordings and their generated files from the page. Avoid
`docker compose down -v` unless you intend to delete **all** jobs and weights.

The standard image uses Debian's [MuseScore 3 package](https://packages.debian.org/bookworm/musescore3)
for native export on both ARM64 and AMD64. It produces MuseScore 3-format `.mscz`
files that MuseScore 4 can import; container engraving/playback may differ from
local MuseScore 4. MusicXML exports are included for interchange. Conversion is
checked during the image build with a real 15-part score, including a check
that no instrument imports as a choir. A job is only marked complete when its
native `.mscz` exists. On a local macOS run, the installed MuseScore 4 is used.

For a single image **including weights**, build the explicit `bundled` target:

```sh
docker build --target bundled -t brass-studio:bundled .
docker run --init --rm -p 127.0.0.1:8000:8000 \
  -v brass-studio-jobs:/data brass-studio:bundled
```

This target downloads and verifies the checkpoints while building. Do not mount
an empty directory over `/models` in that image, since it would hide the bundled
weights. Review the upstream checkpoint redistribution terms described in
“Backend investigation and licenses” before distributing a bundled image.
The image preserves Debian's MuseScore copyright notices under
`/usr/share/doc/musescore3`; its corresponding source package is linked on the
Debian package page. A commercial release needs its dependency and weight
redistribution licenses checked; this packaging does not establish permission
for every upstream component.

### Resources and configuration

Allow Docker at least **8 CPU cores and 16 GB RAM** for the detailed pipeline;
24 GB is preferable. This is CPU inference; Docker on macOS does not use Apple
MPS. Processing time depends on recording length and hardware and can be tens
of minutes. One worker runs one job at a time to avoid loading several large
models concurrently. Pending jobs persist; jobs interrupted by a server restart
are marked failed with retained intermediates and can be uploaded again.
Run **one** web process per data directory; an exclusive worker lock rejects
another process using the same directory.

Compose binds to localhost. This version is a single-user local product with
no account system: everyone who can access its server can view and delete its
jobs. Before exposing it publicly, add authentication, TLS and per-user job
ownership/storage quotas. Do not publish the port directly to the internet.

Environment variables (also editable in `compose.yaml`):

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRASS_DATA` | `web-data` locally, `/data` in Docker | Persistent job storage |
| `BRASS_CACHE` | `.cache/models` locally, `/models` in Docker | Weight cache |
| `BRASS_MAX_UPLOAD_MB` | `100` | Maximum MP3 size; whole request bodies are bounded |
| `BRASS_MAX_DURATION` | `900` | Maximum decoded audio length in seconds |
| `BRASS_MAX_JOBS` | `20` | Maximum stored jobs; delete old jobs when full |
| `BRASS_JOB_TIMEOUT` | `7200` | Maximum seconds for each pipeline stage |
| `BRASS_HOST` | `127.0.0.1` locally, `0.0.0.0` in Docker | Listen address |
| `PORT` | `8000` | HTTP port |
| `MUSESCORE_BIN` | Auto-discovery | Optional native export executable |
| `MUSESCORE_MAJOR` | `4` | Set `3` for a MuseScore 3 exporter |

Local development with the existing weight cache:

```sh
uv sync --python 3.11
uv run brass-web
```

Endpoints are documented at `/docs`: `POST /api/jobs` accepts multipart `file`,
`ensemble` and `passes` (2 or 4); `GET /api/jobs/{id}` exposes job status and
finished artifact names. `/api/health` distinguishes a running server from
model readiness. Scores remain provisional transcriptions; use the included
report and source recording when reviewing musical accuracy.
