#!/usr/bin/env bash
set -euo pipefail
if ! command -v apptainer >/dev/null; then module load apptainer; fi
stage=${1:?expected prepare, job or finish}
shift
case "$stage" in
    prepare) exec bash "$ANGLE_CONTAINER" approach-run --stage prepare --config "$ANGLE_BASE" --output "$ANGLE_OUTPUT" "$@" ;;
    job) exec bash "$ANGLE_CONTAINER" approach-run --stage job --output "$ANGLE_OUTPUT" ;;
    finish) exec bash "$ANGLE_CONTAINER" approach-finish --stage finish --output "$ANGLE_OUTPUT" ;;
    *) exit 2 ;;
esac
