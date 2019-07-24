#!/usr/bin/env python3

"""
Script to run experiments on the Swiss National Supercomputing Centre (CSCS).
https://www.cscs.ch/
"""

import time
import os

txtfile = '''#!/bin/bash -l
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=gpu
#SBATCH --account=sd01
#SBATCH --output=nbody-{0}-%j.log
#SBATCH --error=nbody-{0}-e-%j.log
module load daint-gpu
module load cray-python
module load TensorFlow/1.12.0-CrayGNU-18.08-cuda-9.1-python3
source /users/nperraud/upgan/bin/activate

srun python experiment.py {0}
'''


def launch_simulation(ns):
    sbatch_txt = txtfile.format(ns)
    with open('launch.sh', 'w') as file:
        file.write(sbatch_txt)
    os.system("sbatch launch.sh")
    time.sleep(1)
    os.remove('launch.sh')


if __name__ == '__main__':
    ns = 32
    for _ in range(4):
        print(ns)
        launch_simulation(ns)
        ns = ns*2
        time.sleep(1)
