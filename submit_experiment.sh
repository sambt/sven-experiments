#!/bin/bash
#SBATCH --partition=iaifi_gpu_priority
#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --output=slurm_logs/output-%j.out

source ~/.bash_profile
mamba activate jax
cd /n/home11/sambt/iaifi/sv3/

config=$1
config_path=`dirname $config`
config_name=`basename $config`
shift

python run.py --config-name $config_name "$@"