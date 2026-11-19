#!/usr/bin/env bash
#SBATCH --job-name=pose-basin-analysis
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=pose-basin-analysis-%j.out
#SBATCH --export=ALL
set -euo pipefail
: "${POSE_SOURCE:?POSE_SOURCE required}"
: "${POSE_WORK:?POSE_WORK required}"
: "${POSE_EXPERIMENT:?POSE_EXPERIMENT required}"
if ! command -v apptainer >/dev/null; then module load apptainer; fi
mkdir -p "$POSE_EXPERIMENT/analysis"
# One read-only analysis pass after ALL GPU array tasks completed.
# SLURM --dependency=afterok:<array ID> prevents racing partial generation.
bash "$POSE_SOURCE/experiments/pose_guidance/container.sh" analyse \
  --experiment "$POSE_EXPERIMENT" \
  2>&1 | tee "$POSE_EXPERIMENT/analysis/auto-analysis.log"
