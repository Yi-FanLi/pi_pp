#!/bin/bash
#SBATCH --job-name=pi_pp      #(-J, --job-name)
#SBATCH --account=m3538      #(-A, --account)
#SBATCH --constraint=cpu      #(-A, --account)
#SBATCH --qos=debug            #(-Q, --qos)
#SBATCH --time=00:10:00        #(-t, --time)
#SBATCH --nodes=1              #(-N, --nodes)
#SBATCH --ntasks=8           #(-n, --ntasks)
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1      #(-c, --cpus-per-task)
#SBATCH --mem=256G
##SBATCH --mail-type=begin,end,fail
##SBATCH --mail-user=yifanl@princeton.edu

export SLURM_CPU_BIND="cores"

module load python
#srun -n 8 python xyz2npy.py --natom 864 --id 14 --nsamp 4801
srun -n 8 python zmmp.py --id 14 --nsamp 4801
