#!/bin/bash
#SBATCH -w "GPU6"
source activate tf_gpu
srun python -u ppbm_fold.py  --fold=1 > test.out #1st task
conda deactivate

