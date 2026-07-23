#!/bin/bash
#SBATCH --job-name=SPERR_fft_python
#SBATCH --output=SPERR_fft_%j.out
#SBATCH --ntasks=1
#SBATCH --time=8:00:00          # 运行时间限制 (HH:MM:SS)
#SBATCH --partition=gpucluster       # 队列/分区名称 (根据实际集群修改，如 gpu)
#SBATCH --cpus-per-task=16                 # Number of CPU cores per task

sinfo -o "%n %c %m"  # Shows nodes, their CPU count, and memory
sinfo -N -l  # Detailed node information

# Check NVIDIA GPU status using srun
srun --partition=gpucluster nvidia-smi

# Activate conda
source /Users/923714256/miniconda3/bin/activate
conda activate grandlib

# =============================================================================
# Run the Python script
# =============================================================================

echo "Starting 4 parallel tasks on 4 GPUs..."
CUDA_VISIBLE_DEVICES=0 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task nyx_b --save_recons &
CUDA_VISIBLE_DEVICES=1 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task nyx_t --save_recons &
CUDA_VISIBLE_DEVICES=2 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task nyx_d --save_recons &
CUDA_VISIBLE_DEVICES=3 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task miranda --save_recons &
wait

echo "Starting remaining 2 tasks..."
CUDA_VISIBLE_DEVICES=0 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task warpx --save_recons &
CUDA_VISIBLE_DEVICES=1 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task mag --save_recons &
wait

echo "All training finished, now generating plot..."
CUDA_VISIBLE_DEVICES=0 python /Users/923714256/Final_design_for_BO_adaptive_compression/SPERR/SPERR_fft.py --task plot

echo "Python script finished!"
