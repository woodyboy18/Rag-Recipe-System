#!/bin/bash
#PBS -N Rag-Recipe
#PBS -l nodes=1:ncpus=10
#PBS -l walltime=48:00:00
#PBS -q Gpu1-10g
#PBS -o rag.out
#PBS -e rag.err

echo "===== JOB START ====="
date
hostname

cd /storage/home/bagler/sonukashif/Rag-Recipe-System || exit 1

echo "Current Directory:"
pwd

source /storage/home/bagler/bk/anaconda3/etc/profile.d/conda.sh

echo "Conda sourced"

conda activate rag

echo "Environment activated"

which python
python --version

echo "CUDA_VISIBLE_DEVICES before export:"
echo $CUDA_VISIBLE_DEVICES

export CUDA_VISIBLE_DEVICES=MIG-f3d88dc3-260f-516c-9565-e895709806c8

echo "CUDA_VISIBLE_DEVICES after export:"
echo $CUDA_VISIBLE_DEVICES

echo "Running interactive_rag.py"

python scripts/interactive_rag.py

echo "Python Exit Code: $?"

echo "===== JOB END ====="
