#!/bin/bash
#SBATCH --job-name=VAE5
#SBATCH --output=out_job_%j.out
#SBATCH --error=err_job_%j.err
#SBATCH --partition=ga80-1gpu

#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1


# Activate the virtual environment (if applicable)
# source /path/to/venv/bin/activate
source /home/iwasakidk/env_name/bin/activate
# Change to the directory where the Python script is located
cd /home/iwasakidk/VAE

# Run the Python script using mpiexec and passing any necessary command line arguments
mpiexec -n 1 /home/iwasakidk/env_name/bin/python3 VAE_train_beta_01.py
