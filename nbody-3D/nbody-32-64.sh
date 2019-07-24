#!/bin/bash -l
#SBATCH --time=23:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=sd01
#SBATCH --output=nbody-64-%j.log
#SBATCH --error=nbody-64-e-%j.log

module load daint-gpu
module load cray-python
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3

source /users/nperraud/upgan/bin/activate

srun python nbody-32-64.py
