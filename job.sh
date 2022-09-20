#!/bin/bash

#!/bin/bash

#SBATCH --job-name=mfp
#SBATCH --output=mfp-%a-%A.out
#SBATCH --array=54
# SBATCH --array=0-81%5
#SBATCH --partition=sched_mit_mki,sched_mit_mvogelsb,sched_mit_mki_preempt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --constraint=centos7
#SBATCH --mem-per-cpu=6000 # 6GB of memory per CPU
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=arsmith@mit.edu

## Module setup
. /etc/profile.d/modules.sh
module load gcc/9.3.0 openmpi/4.0.5

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export HDF5_USE_FILE_LOCKING=FALSE

mpirun --mca opal_warn_on_missing_libcuda 0 --mca btl '^openib' --mca psm2 ucx -np $SLURM_NTASKS \
  ./thesan-mfp /nfs/mvogelsblab001/Thesan/Thesan-1/postprocessing/smooth_renderings/smooth_renderings_1cMpc output ${SLURM_ARRAY_TASK_ID}


