#!/bin/bash
#SBATCH --job-name=h2o8b         # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=8              # total number of tasks across all nodes
#SBATCH --tasks-per-node=8
##SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
##SBATCH --gres=gpu:0             # number of gpus per node
#SBATCH --time=00:59:00          # total run time limit (HH:MM:SS)
##SBATCH --mail-type=begin        # send email when job begins
##SBATCH --mail-type=end          # send email when job ends
##SBATCH --mail-user=yifanl@princeton.edu

module purge
module load anaconda3/2020.11
conda activate /scratch/gpfs/yifanl/usr/licensed/anaconda3/2021.11/dpdev6

### firstly run this command
srun --mpi=pmi2 -n 8 python xyz2npy.py --natom 50 --nsamp 20001

### secondly run this command
srun --mpi=pmi2 -n 8 python fnm2f.py

### thirdly run this command
srun --mpi=pmi2 -n 8 python ke_estimator.py
