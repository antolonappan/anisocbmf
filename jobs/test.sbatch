#!/bin/bash
# Small end-to-end test: 4 MPI ranks, 4 realizations, 2 configurations.
# Writes to a separate test libdir so production caches stay clean.
#SBATCH --job-name=acb_test
#SBATCH --partition=gen
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/home/alonappan/ceph/mfproject/logs/%x_%j.log

set -euo pipefail
module purge
source /mnt/home/alonappan/anaconda3/etc/profile.d/conda.sh
conda activate base
# Each rank runs its spherical-harmonic transforms on its own cpus-per-task cores.
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK DUCC0_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /mnt/home/alonappan/workspace/anisocbmf
mpirun -np "$SLURM_NTASKS" --map-by "slot:PE=$SLURM_CPUS_PER_TASK" --bind-to core python src/run_sims.py \
    --libdir /mnt/home/alonappan/ceph/mfproject/test \
    --nsims 4 --acb 1e-6 --alpha None 0.3 \
    --mask data/sky1.fits
