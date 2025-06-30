#!/bin/bash

#SBATCH -o ./logs/launch_hparam_search_batch_size_tonsilitis.out

#SBATCH -e ./logs/launch_hparam_search_batch_size_tonsilitis.err

#SBATCH -J launch_hparam_search_batch_size_tonsilitis

#SBATCH -p cpu_p

#SBATCH --qos=cpu_normal

#SBATCH -c 4

#SBATCH --mem=90G

#SBATCH -t 1-00:00

#SBATCH --nice=10000

export HYDRA_FULL_ERROR=1

conda activate sc_exp_design 

cd /home/icb/alessandro.palma/environment/scportrait_ot/src

python main.py --multirun datamodule=tonsilitis hparam_search=sweep_batch_size logger.project=scportrait_ot_sweep_batch_size_tonsilitis