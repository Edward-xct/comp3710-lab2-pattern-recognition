#!/bin/bash
#SBATCH --job-name=cuda-check
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu
#SBATCH --time=00:05:00
#SBATCH --output=cuda_check_%j.out
#SBATCH --error=cuda_check_%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/bin/activate"
conda activate torch

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"
nvidia-smi
python test_cuda.py
