#!/usr/bin/env bash
# Explicit three-arm surface-elevation study from a saved NOMINEE antigen setup.
set -euo pipefail

usage() {
    cat <<'HELP'
Usage: launch_surface_elevation.sh ANTIGEN_SETUP.yaml [options]

Prompts for chain-qualified AUTHOR residues unless --residues is provided.
Runs ordinary targeted, legacy placement-ray tilt, and surface-elevation arms.

  --residues 'A:10,12-15;B:7'  Replace the setup's epitope (AUTHOR numbering)
  --antigen-cif FILE           Override the setup's antigen coordinates
  --nanobody-setup FILE        Use a saved VHH framework/CDR setup
  --output NEW_DIRECTORY      Fresh study directory (default: unique scratch dir)
  --elevation MIN MAX         Signed surface elevation, default 70 90 degrees
  --legacy-tilt MIN MAX       Legacy placement-ray tilt, default 0 20 degrees
  --three-arms                Explicitly name the default matched three-arm mode
  --gfp-pair-ablation         GFP-only ordinary/4/8/12/16 pairs, budget per arm
  --backbones N               Total requested parents PER ARM, default 1000
  --replicates R              Matched replicate blocks, default 10
  --seed N                    Initial replicate seed, default 0
  --sequences-per-parent N    Inverse-folded sequences per parent, default 1
  --inverse-folder NAME       Default solublempnn; same setting for every arm
  --recycling-steps N         Default 5
  --diffusion-steps N         Default 200
  --preview                   CPU geometry preparation; no submission or MSAs

The installed native refold emits five samples per designed sequence. Without
--nanobody-setup, use the shipped caplacizumab/7eow framework with CDR lengths
6-8, 6-8 and 8-20. No checkpoint weights or biological validation are bundled.
Use launch_approach.sh for historical --angle/--paired commands; --angle has
never meant surface elevation and is deliberately rejected here.
HELP
}

fail() { echo "ERROR: $*" >&2; exit 2; }
require_value() { [[ $# -ge 2 && -n "$2" ]] || fail "Missing value for $1"; }
check_bind_path() {
    case "$1" in *:*|*,*) fail "Path contains an Apptainer bind separator: $1" ;; esac
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 ]] || { usage >&2; exit 2; }
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
base=$(realpath -e -- "$1")
[[ -f "$base" ]] || fail "Antigen setup is not a file: $base"
shift
residues=""
nano="-"
antigen=""
output=""
elevation=(70 90)
legacy_tilt=(0 20)
backbones=1000
replicates=10
seed=0
sequences=1
inverse_folder=solublempnn
recycling=5
diffusion=200
preview=false
gfp_ablation=false
extra=()

while (( $# )); do
    case "$1" in
        --residues) require_value "$@"; residues=$2; shift 2 ;;
        --nanobody-setup) require_value "$@"; nano=$(realpath -e -- "$2"); shift 2 ;;
        --antigen-cif) require_value "$@"; antigen=$(realpath -e -- "$2"); shift 2 ;;
        --output) require_value "$@"; output=$(realpath -m -- "$2"); shift 2 ;;
        --elevation|--legacy-tilt)
            [[ $# -ge 3 ]] || fail "$1 needs two endpoints"
            if [[ "$1" == --elevation ]]; then elevation=("$2" "$3"); else legacy_tilt=("$2" "$3"); fi
            shift 3 ;;
        --backbones) require_value "$@"; backbones=$2; shift 2 ;;
        --replicates) require_value "$@"; replicates=$2; shift 2 ;;
        --seed) require_value "$@"; seed=$2; shift 2 ;;
        --sequences-per-parent) require_value "$@"; sequences=$2; shift 2 ;;
        --inverse-folder) require_value "$@"; inverse_folder=$2; shift 2 ;;
        --recycling-steps) require_value "$@"; recycling=$2; shift 2 ;;
        --diffusion-steps) require_value "$@"; diffusion=$2; shift 2 ;;
        --three-arms) shift ;;
        --gfp-pair-ablation) gfp_ablation=true; shift ;;
        --preview) preview=true; shift ;;
        --standoff|--weights|--model-config)
            require_value "$@"; extra+=("$1" "$2"); shift 2 ;;
        --angle) fail "--angle retains legacy tilt semantics; use --elevation and --legacy-tilt here" ;;
        --help|-h) usage; exit 0 ;;
        *) fail "Unknown option: $1 (see --help)" ;;
    esac
done
for value in "$backbones" "$replicates" "$sequences" "$recycling" "$diffusion"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "Counts must be positive integers"
done
[[ "$seed" =~ ^[0-9]+$ ]] || fail "Seed must be a nonnegative integer"
(( replicates <= backbones )) || fail "Replicates cannot exceed parents per arm"
if [[ -z "$residues" ]]; then
    printf 'Target AUTHOR residues (e.g. A:10,12-15;B:7): ' >&2
    IFS= read -r residues || fail "No residue selection received; use --residues for noninteractive submission"
fi
[[ -n "${residues//[[:space:]]/}" ]] || fail "A target residue selection is required"
if [[ -z "$output" ]]; then
    work_parent="${ANGLE_WORK_DIR:-/vast/scratch/users/$USER}"
    mkdir -p -- "$work_parent"
    output=$(mktemp -d "$(realpath -- "$work_parent")/surface-elevation-XXXXXXXX")
else
    mkdir -p -- "$(dirname -- "$output")"
    mkdir -- "$output" || fail "Choose a fresh output directory: $output"
fi
for path in "$base" "$output"; do check_bind_path "$path"; done
export POSE_WORK="$output" ANGLE_TASK_DIR="$(dirname -- "$base")"
if ! command -v apptainer >/dev/null; then module load apptainer; fi

setup_args=("$base" "$nano" "$residues" "$output" --surface-study
    --defaults /opt/angle-defaults.yaml --sequences-per-parent "$sequences"
    --inverse-folder "$inverse_folder" --recycling-steps "$recycling" --diffusion-steps "$diffusion")
for path in "$nano" "$antigen"; do
    [[ -z "$path" || "$path" == - ]] && continue
    [[ -f "$path" ]] || fail "Setup input is not a file: $path"
    check_bind_path "$path"
    export POSE_EXTRA_BINDS="${POSE_EXTRA_BINDS:+$POSE_EXTRA_BINDS,}$path:$path:ro"
done
if [[ -n "$antigen" ]]; then setup_args+=(--antigen-cif "$antigen"); fi
refs=$(bash "$here/container.sh" approach-setups "${setup_args[@]}" --list-inputs)
while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    [[ -f "$path" ]] || fail "Missing setup input: $path"
    check_bind_path "$path"
    export POSE_EXTRA_BINDS="${POSE_EXTRA_BINDS:+$POSE_EXTRA_BINDS,}$path:$path:ro"
done <<< "$refs"
bash "$here/container.sh" approach-setups "${setup_args[@]}"
base="$output/inputs/targeted.json"
options=(--elevation "${elevation[@]}"
    --backbones "$backbones" --replicates "$replicates" --seed "$seed" --auto-msa "${extra[@]}")
echo "Run directory: $output"
if [[ "$gfp_ablation" == true ]]; then
    options+=(--gfp-pair-ablation)
    echo "GFP-only ablation: 0 | 4 | 8 | 12 | 16 pairs; $((backbones * 5)) total requested parents"
else
    options+=(--three-arms --legacy-tilt "${legacy_tilt[@]}")
    echo "Arms: ordinary targeted | legacy tilt ${legacy_tilt[*]} deg | surface elevation ${elevation[*]} deg"
fi
echo "Requested parents: $backbones per arm, $replicates matched replicates"
echo "Sequence budget: $sequences per parent; five unrestrained native refolds per sequence"
if [[ "$preview" == true ]]; then
    exec bash "$here/container.sh" approach-preview --config "$base" --output "$output" "${options[@]}"
fi
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    exec bash "$here/container.sh" approach-run --config "$base" --output "$output" "${options[@]}"
fi

export ANGLE_BASE="$base" ANGLE_OUTPUT="$output" ANGLE_CONTAINER="$here/container.sh"
gpu=(--partition="${ANGLE_PARTITION:-gpuq}" --gres="${ANGLE_GRES:-gpu:A30:1}"
    --cpus-per-task="${ANGLE_CPUS:-4}" --mem="${ANGLE_MEMORY:-48G}" --time="${ANGLE_TIME:-24:00:00}")
submit() {
    local id
    id=$(sbatch --parsable --ntasks=1 --output="$output/slurm-%x-%A_%a.log" "$@")
    id=${id%%;*}
    [[ "$id" =~ ^[0-9]+$ ]] || fail "Unexpected sbatch receipt: $id"
    echo "$id"
}
count=$(bash "$here/container.sh" approach-finish --layout-only --config "$base" --output "$output" "${options[@]}" | tail -n 1)
[[ "$count" =~ ^[1-9][0-9]*$ ]] || fail "Invalid array size: $count"
driver="$here/approach_stage.sh"
prepare=$(submit --job-name=elevation-prepare "${gpu[@]}" "$driver" prepare "${options[@]}")
echo "Preparation: $prepare"
array=$(submit --job-name=elevation-design "${gpu[@]}" --kill-on-invalid-dep=yes \
    --dependency="afterok:$prepare" --array="0-$((count-1))%${ANGLE_CONCURRENCY:-4}" "$driver" job)
echo "Design array: $array"
finish=$(submit --job-name=elevation-report --partition="${ANGLE_CPU_PARTITION:-regular}" \
    --cpus-per-task=2 --mem=12G --time="${ANGLE_REPORT_TIME:-08:00:00}" \
    --dependency="afterany:$array" "$driver" finish)
echo "Analysis job: $finish"
echo "Report: $output/analysis/report.html"
