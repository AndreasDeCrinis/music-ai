#!/bin/sh
# Measure how many transcription passes help, on the GPU container.
#
# usage: scripts/gpu-benchmark.sh NAME [PASSES]
#   bench/NAME.mp3  audio rendered from a score (e.g. MuseScore: mscore -o NAME.mp3 score.mscz)
#   bench/NAME.mid  MIDI export of the same score (mscore -o NAME.mid score.mscz)
# Finished steps are skipped, so the script can be re-run after an interruption.
set -eu
name=$1
passes=${2:-9}
compose="docker compose -f compose.yaml -f compose.gpu.yaml"
[ -f "bench/$name.mp3" ] && [ -f "bench/$name.mid" ] || {
  echo "missing bench/$name.mp3 or bench/$name.mid" >&2
  exit 1
}
chmod a+rwx bench
$compose run --rm -v "$PWD/bench:/bench" app sh -ec "
  base=/bench/$name-base
  run=/bench/$name-q$passes
  [ -f \$base/diagnostics/transcription_report.json ] || brass-transcribe /bench/$name.mp3 \
    --ensemble brassaranka --cache /models --output \$base --no-mscz
  [ -f \$run/diagnostics/transcription_report.json ] || brass-transcribe improve \$base \
    --ensemble brassaranka --cache /models --passes $passes --output \$run --no-mscz
  brass-transcribe benchmark \$run /bench/$name.mid
"
