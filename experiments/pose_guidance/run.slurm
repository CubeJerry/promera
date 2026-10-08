#!/usr/bin/env bash
#SBATCH --job-name=pose-guide
#SBATCH --partition=gpuq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=pose-guide-%A_%a.out
#SBATCH --export=ALL
set -euo pipefail
: "${POSE_SOURCE:?Set POSE_SOURCE to the experimental Promera checkout}"
: "${POSE_EXPERIMENT:?Set POSE_EXPERIMENT to the prepared experiment directory}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as an array job}"
if ! command -v apptainer >/dev/null; then module load apptainer; fi
bash "$POSE_SOURCE/experiments/pose_guidance/container.sh" run \
    --experiment "$POSE_EXPERIMENT" --job "$SLURM_ARRAY_TASK_ID"
