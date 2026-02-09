#!/bin/bash
#SBATCH --job-name=llava_inf
#SBATCH --cpus-per-task=4
#SBATCH --qos=short
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --constraint="ampere&gmem80"

# Generate random 4-digit ID for this run
RANDOM_ID=$(printf "%04d" $((RANDOM % 10000)))

# Create logs folder if it doesn't exist
mkdir -p logs

# Redirect output and error to log files
exec > logs/log_${RANDOM_ID}.txt 2> logs/error_${RANDOM_ID}.txt

# Print job info
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Log ID: $RANDOM_ID"
echo "================================"

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venvCB

# Run Python in unbuffered mode to see prints in real-time
python -u main.py \
    --input data/bench_full_fidel.jsonl \
    --images ../../../../share/datasets/spatial_bench_dataset/images \
    --model llava-hf/llava-onevision-qwen2-7b-ov-hf

# Print completion info
echo "================================"
echo "Job completed at: $(date)"
