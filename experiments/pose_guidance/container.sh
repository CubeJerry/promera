#!/usr/bin/env bash
# Load only this experimental task. Keep the patched image's Promera/Boltz-IF code.
set -euo pipefail
: "${POSE_WORK:?Set POSE_WORK to a separate writable experiment workspace}"
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
WORK=$(cd -- "$POSE_WORK" && pwd -P)
# NOMINEE config specifies installation_parent; current image/asset releases live
# beneath <installation_parent>/.nominee. Manual overrides remain supported.
if [[ -z "${PROMERA_IMAGE:-}" || -z "${PROMERA_ASSETS:-}" ]]; then
    config="${POSE_NOMINEE_CONFIG:-${NOMINEE_HOST_CONFIG:-${NOMINEE_CONFIG:-}}}"
    [[ -n "$config" ]] || config="${HOME:?HOME must be set}/dev/config.yaml"
    [[ -f "$config" ]] || {
        echo "ERROR: NOMINEE config not found: $config (set NOMINEE_CONFIG or both PROMERA_IMAGE and PROMERA_ASSETS)" >&2
        exit 2
    }
    discovered=$(python3 "$HERE/nominee_paths.py" "$config") || exit 2
    mapfile -t managed_paths <<< "$discovered"
    [[ ${#managed_paths[@]} -eq 2 ]] || {
        echo "ERROR: NOMINEE path resolver returned invalid paths: $config" >&2
        exit 2
    }
    PROMERA_IMAGE="${PROMERA_IMAGE:-${managed_paths[0]}}"
    PROMERA_ASSETS="${PROMERA_ASSETS:-${managed_paths[1]}}"
fi
[[ -f "$PROMERA_IMAGE" ]] || {
    echo "ERROR: active Promera image not found: $PROMERA_IMAGE (check NOMINEE installation)" >&2
    exit 2
}
[[ -d "$PROMERA_ASSETS" ]] || {
    echo "ERROR: active Promera assets not found: $PROMERA_ASSETS (check NOMINEE installation)" >&2
    exit 2
}
IMAGE=$(readlink -f -- "$PROMERA_IMAGE")
ASSETS=$(readlink -f -- "$PROMERA_ASSETS")
[[ -f "$ASSETS/checkpoints/promera_2606.ckpt" ]] || {
    echo "ERROR: Promera checkpoint missing beneath $ASSETS" >&2
    exit 2
}
printf 'Promera image: %s\nPromera assets: %s\n' "$IMAGE" "$ASSETS" >&2
[[ ! -d "$WORK/promera" ]] || { echo 'ERROR: POSE_WORK must not be a Promera source checkout' >&2; exit 2; }
# Use Apptainer's supported --home mount below, not --env HOME=...:
# Apptainer forbids APPTAINERENV_HOME overrides and otherwise leaves the
# read-only NSS home (/home/users/...) in place. Keep all writes in scratch.
mkdir -p \
    "$WORK/.home" "$WORK/.config" "$WORK/.local/share" "$WORK/.tmp" \
    "$WORK/.cache/torch" "$WORK/.cache/torchinductor" \
    "$WORK/.cache/triton" "$WORK/.cache/nvidia" \
    "$WORK/.cache/huggingface/hub" "$WORK/.cache/huggingface/transformers" "$WORK/.cache/matplotlib" \
    "$WORK/.cache/numba" "$WORK/.cache/cupy"
chmod 700 "$WORK/.home" "$WORK/.config" "$WORK/.local/share"
TEMP=$(mktemp -d "$WORK/.tmp/pose-guide.XXXXXX")
mkdir -p "$TEMP/xdg-runtime"
chmod 700 "$TEMP/xdg-runtime"
trap 'rm -rf -- "$TEMP"' EXIT

# Latest NOMINEE keeps prepared run schemas and MSA caches under the managed
# installation parent, outside the pose experiment's writable output directory.
# Make the parent readable but keep the experiment workspace separately writable.
binds="$HERE:/opt/pose_guidance:ro"
if [[ -n "${discovered:-}" ]]; then
    parent="${managed_paths[0]%/.nominee/images/promera/current.sif}"
    [[ -n "$parent" && "$parent" != "${managed_paths[0]}" && -d "$parent" ]] || {
        echo "ERROR: could not resolve managed installation parent from $config" >&2
        exit 2
    }
    binds+=",$parent:$parent:ro"
    if [[ -d "$parent/.nominee/cache" ]]; then
        binds+=",$parent/.nominee/cache:/opt/nominee/cache:ro"
    fi
fi
binds+=",$WORK:$WORK:rw,$ASSETS:/opt/nominee/assets:ro"
if [[ -d "$ASSETS/ligandmpnn/model_params" ]]; then
    binds+=",$ASSETS/ligandmpnn/model_params:/opt/nominee/component/LigandMPNN/model_params:ro"
fi
if [[ -n "${POSE_EXTRA_BINDS:-}" ]]; then binds+=",$POSE_EXTRA_BINDS"; fi
# --home SRC:DEST sets the container's home to the scratch-backed directory.
# Do not disable the home mount with --no-home or --no-mount home.
# The working directory is explicitly bound; suppress the implicit CWD mount.
args=(exec --cleanenv --no-mount cwd --home "$WORK/.home:$WORK/.home" --bind "$binds" --pwd "$WORK")
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
# HOME is a reserved Apptainer variable: use --home above, never --env HOME.
# The launched process also gets an explicit HOME via /usr/bin/env (not
# APPTAINERENV_HOME), ensuring Python and any subprocesses see writable scratch.
args+=(--env "XDG_CONFIG_HOME=$WORK/.config")
args+=(--env "XDG_CACHE_HOME=$WORK/.cache" --env "XDG_DATA_HOME=$WORK/.local/share")
args+=(--env "XDG_RUNTIME_DIR=$TEMP/xdg-runtime" --env "TMPDIR=$TEMP")
args+=(--env "TORCH_HOME=$WORK/.cache/torch")
args+=(--env "TORCHINDUCTOR_CACHE_DIR=$WORK/.cache/torchinductor")
args+=(--env "TRITON_CACHE_DIR=$WORK/.cache/triton")
args+=(--env "CUDA_CACHE_PATH=$WORK/.cache/nvidia")
args+=(--env "HF_HOME=$WORK/.cache/huggingface")
args+=(--env "HUGGINGFACE_HUB_CACHE=$WORK/.cache/huggingface/hub")
args+=(--env "HF_HUB_CACHE=$WORK/.cache/huggingface/hub")
args+=(--env "TRANSFORMERS_CACHE=$WORK/.cache/huggingface/transformers")
args+=(--env "MPLCONFIGDIR=$WORK/.cache/matplotlib" --env "MPLBACKEND=Agg")
args+=(--env "NUMBA_CACHE_DIR=$WORK/.cache/numba")
args+=(--env "CUPY_CACHE_DIR=$WORK/.cache/cupy")
args+=(--env 'PROMERA_RECORD_TRAJECTORY=0' --env 'PYTHONNOUSERSITE=1' --env 'OMP_NUM_THREADS=1')
apptainer "${args[@]}" "$IMAGE" /usr/bin/env "HOME=$WORK/.home" python /opt/pose_guidance/experiment.py "$@"
