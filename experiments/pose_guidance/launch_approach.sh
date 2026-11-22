#!/usr/bin/env bash
# Host-side launcher for NOMINEE's existing Promera image and weights.
set -euo pipefail
usage() {
    echo "Usage: $0 BASE_TASK_YAML NEW_OUTPUT_DIR [--angle MIN MAX] [--paired] [--backbones N] [--replicates N] [--preview] [--target-structure CIF]" >&2
}
if [[ "${1:-}" == --help ]]; then usage; exit 0; fi
if [[ $# -lt 2 ]]; then usage; exit 2; fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base=$(realpath -e -- "$1")
output=$(realpath -m -- "$2")
shift 2
mkdir -p -- "$(dirname -- "$output")"
mkdir -- "$output"
export POSE_WORK="$output"
export ANGLE_TASK_DIR="$(dirname -- "$base")"
action=approach-run
for arg in "$@"; do
    if [[ "$arg" == --preview ]]; then action=approach-preview; fi
done
runner=(bash "$here/container.sh" "$action" --config "$base" --output "$output" "$@")
if [[ "$action" == approach-preview || -n "${SLURM_JOB_ID:-}" ]]; then
    if ! command -v apptainer >/dev/null; then module load apptainer; fi
    exec "${runner[@]}"
fi
# Prepare the plan once. Each replicate/arm then runs in its own allocation.
# Query the resolved config inside the same image to size the array.
export ANGLE_BASE="$base" ANGLE_OUTPUT="$output" ANGLE_CONTAINER="$here/container.sh"
gpu=(--partition="${ANGLE_PARTITION:-gpuq}" --gres="${ANGLE_GRES:-gpu:A30:1}"
     --cpus-per-task="${ANGLE_CPUS:-4}" --mem="${ANGLE_MEMORY:-48G}" --time="${ANGLE_TIME:-24:00:00}")
submit() {
    local id
    id=$(sbatch --parsable --ntasks=1 --output="$output/slurm-%x-%A_%a.log" "$@")
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch receipt: $id" >&2; return 1; }
    echo "$id"
}
# Use actual script files: Slurm --wrap runs /bin/sh and cannot preserve Bash arrays.
driver="$here/approach_stage.sh"
if ! command -v apptainer >/dev/null; then module load apptainer; fi
count=$(bash "$here/container.sh" approach-finish --layout-only --config "$base" --output "$output" "$@" | tail -n 1)
[[ "$count" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid array size: $count" >&2; exit 2; }
prepare=$(submit --job-name=angle-prepare "${gpu[@]}" "$driver" prepare "$@")
echo "Preparation: $prepare"
array=$(submit --job-name=angle-design "${gpu[@]}" --kill-on-invalid-dep=yes \
    --dependency="afterok:$prepare" --array="0-$((count-1))%${ANGLE_CONCURRENCY:-4}" "$driver" job)
echo "Design array: $array"
finish=$(submit --job-name=angle-report --partition="${ANGLE_CPU_PARTITION:-regular}" \
    --cpus-per-task=2 --mem=12G --time="${ANGLE_REPORT_TIME:-08:00:00}" \
    --dependency="afterany:$array" "$driver" finish)
echo "Analysis job: $finish"
echo "Report: $output/analysis/summary.json"
