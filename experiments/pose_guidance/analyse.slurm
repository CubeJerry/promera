#!/usr/bin/env bash
#SBATCH --job-name=pose-guide-analyse
#SBATCH --partition=regular
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=pose-guide-analysis-%j.out
#SBATCH --export=ALL
set -euo pipefail

: "${POSE_SOURCE:?Set POSE_SOURCE to the experimental Promera checkout}"
: "${POSE_EXPERIMENT:?Set POSE_EXPERIMENT to the prepared experiment directory}"

# This job is scheduled with afterany on the entire GPU array, so even partial
# or failed runs produce diagnostic CSVs. The analyzer exits nonzero for an
# incomplete experiment and writes analysis/status.json accordingly.
if ! command -v apptainer >/dev/null; then module load apptainer; fi
bash "$POSE_SOURCE/experiments/pose_guidance/container.sh" analyse \
    --experiment "$POSE_EXPERIMENT"
