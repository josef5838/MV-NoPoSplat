#!/bin/bash
#SBATCH --job-name=F_3
#SBATCH --chdir=/cluster/home/yihuli/MV-NoPoSplat
#SBATCH --output=./sbatch_log/F_1_%j.out
#SBATCH --ntasks=1
#SBATCH --nodes=1

#SBATCH --gpus=rtx_4090:1
#SBATCH --mem-per-cpu=12g
#SBATCH --gres=gpumem:24g
#SBATCH --time=2-1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=yihua.li@uzh.ch

nvidia-smi

# Load modules if needed
source /cluster/home/yihuli/miniconda3/etc/profile.d/conda.sh
# Activate conda
conda activate /cluster/project/cvg/students/mv-nopo/noposplat-env  # Or mamba activate if using Mamba

python -m src.main +experiment=re10k_fast3r_1x1 wandb.mode=online wandb.name=re10k
