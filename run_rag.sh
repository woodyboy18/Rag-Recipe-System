#!/bin/bash
#PBS -N RAG-JOB
#PBS -l nodes=1:ncpus=10
#PBS -l walltime=48:00:00
#PBS -q Gpu1-10g
#PBS -o rag.out
#PBS -e rag.err

cd $PBS_O_WORKDIR

source ~/.bashrc
conda activate rag

python scripts/interactive_rag.py
