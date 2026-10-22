#!/bin/bash
# 100 realizations, one per MPI rank (4 threads each), for every combination of
# Acb in {1e-7, 1e-6} and alpha in {None, 0.3, 0.2, 0.1, 0.05, 0.001, 0.0001} deg.
# Experiment: 2 uK-arcmin polarization noise, 5 arcmin FWHM, mask data/sky1.fits (f_sky = 0.70).
# Restartable: finished realizations are skipped when resubmitted.
#SBATCH --job-name=acb_prod
#SBATCH --partition=gen
#SBATCH --ntasks=100
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
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
    --libdir /mnt/home/alonappan/ceph/mfproject \
    --nsims 100 \
    --acb 1e-7 1e-6 \
    --alpha None 0.3 0.2 0.1 0.05 0.001 0.0001 \
    --nlevp 2 --fwhm 5 \
    --mask data/sky1.fits
