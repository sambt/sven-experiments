#!/bin/bash
#SBATCH --partition=gpu_test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=diag_env
#SBATCH --output=slurm_logs/diag_env-%j.out
cd /n/home11/sambt/iaifi/sv3
echo "OMP_NUM_THREADS=${OMP_NUM_THREADS-<unset>}  MKL_NUM_THREADS=${MKL_NUM_THREADS-<unset>}  SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK  SLURM_CPUS_ON_NODE=$SLURM_CPUS_ON_NODE"
echo "nproc=$(nproc)  nproc --all=$(nproc --all)"; grep Cpus_allowed_list /proc/self/status
env | grep -i "omp\|mkl\|openblas\|numexpr" | head
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
.venv/bin/python -c "import os,torch; print('affinity cores:',len(os.sched_getaffinity(0)),'torch.get_num_threads():',torch.get_num_threads())"
env -u OMP_NUM_THREADS .venv/bin/python -c "import os,torch; print('with OMP unset -> torch.get_num_threads():',torch.get_num_threads())"
