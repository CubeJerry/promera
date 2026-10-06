#!/usr/bin/env bash
# Load only this experimental task. Keep the patched image's Promera/Boltz-IF code.
set -euo pipefail
: "${PROMERA_IMAGE:?Set PROMERA_IMAGE to the existing patched Promera .sif}"
: "${PROMERA_ASSETS:?Set PROMERA_ASSETS to the directory containing checkpoints/, tinyprot/, and boltzgen/}"
: "${POSE_WORK:?Set POSE_WORK to a separate writable experiment workspace}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WORK=$(cd -- "$POSE_WORK" && pwd -P)
ASSETS=$(cd -- "$PROMERA_ASSETS" && pwd -P)
IMAGE=$(readlink -f -- "$PROMERA_IMAGE")
[[ -f "$IMAGE" ]] || { echo 'ERROR: Promera image not found' >&2; exit 2; }
[[ -f "$ASSETS/checkpoints/promera_2606.ckpt" ]] || { echo 'ERROR: incorrect asset root (checkpoint missing)' >&2; exit 2; }
[[ ! -d "$WORK/promera" ]] || { echo 'ERROR: POSE_WORK must not be a Promera source checkout' >&2; exit 2; }
mkdir -p "$WORK/.cache/torch" "$WORK/.cache/triton" "$WORK/.tmp"
TEMP=$(mktemp -d "$WORK/.tmp/pose-guide.XXXXXX")
trap 'rm -rf -- "$TEMP"' EXIT
binds="$HERE:/opt/pose_guidance:ro,$WORK:$WORK,$ASSETS:/opt/nominee/assets:ro"
if [[ -d "$ASSETS/ligandmpnn/model_params" ]]; then
    binds+=",$ASSETS/ligandmpnn/model_params:/opt/nominee/component/LigandMPNN/model_params:ro"
fi
if [[ -n "${POSE_EXTRA_BINDS:-}" ]]; then binds+=",$POSE_EXTRA_BINDS"; fi
args=(exec --cleanenv --bind "$binds" --pwd "$WORK")
# CPU preparation and analysis do not require a GPU allocation.
if [[ "${1:-}" == run ]]; then args+=(--nv); fi
for var in CUDA_VISIBLE_DEVICES SLURM_JOB_ID SLURM_ARRAY_TASK_ID SLURM_CPUS_PER_TASK SLURM_NTASKS_PER_NODE SLURM_NNODES SLURM_NTASKS SLURM_PROCID SLURM_LOCALID SLURM_NODEID; do
    if [[ -n "${!var:-}" ]]; then args+=(--env "$var=${!var}"); fi
done
args+=(--env "POSE_RUNTIME_IMAGE=$IMAGE")
if commit=$(git -C "$HERE" rev-parse HEAD 2>/dev/null); then args+=(--env "POSE_EXPERIMENT_COMMIT=$commit"); fi
args+=(--env "PYTHONPATH=/opt/pose_guidance:/opt/nominee/component/promera:/opt/nominee/repo")
args+=(--env "PROMERA_WEIGHTS=/opt/nominee/assets/checkpoints/promera_2606.ckpt")
args+=(--env "TINYPROT_CACHE=/opt/nominee/assets/tinyprot")
args+=(--env "NOMINEE_ASSET_ROOT=/opt/nominee/assets")
args+=(--env "ABMPNN_CHECKPOINT=/opt/nominee/assets/checkpoints/abmpnn.pt")
args+=(--env "LIGANDMPNN_DIR=/opt/nominee/component/LigandMPNN")
args+=(--env "TMPDIR=$TEMP" --env "TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/torch")
args+=(--env "TRITON_CACHE_DIR=$WORK/.cache/triton" --env 'PROMERA_RECORD_TRAJECTORY=0')
args+=(--env 'PYTHONNOUSERSITE=1' --env 'OMP_NUM_THREADS=1')
apptainer "${args[@]}" "$IMAGE" python /opt/pose_guidance/experiment.py "$@"
