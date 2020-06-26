#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:2
#SBATCH --mem=16000  # memory in Mb
#SBATCH -o outfile_1 # send stdout to sample_experiment_outfile
#SBATCH -e errfile_1  # send stderr to sample_experiment_errfile
#SBATCH -t 8:00:00  # time requested in hour:minute:secon
export CUDA_HOME=/opt/cuda-8.0.44

export CUDNN_HOME=/opt/cuDNN-6.0_8.0

export STUDENT_ID=s1027418

export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH

export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH

export CPATH=${CUDNN_HOME}/include:$CPATH

export PATH=${CUDA_HOME}/bin:${PATH}

export PYTHON_PATH=$PATH

mkdir -p /disk/scratch/${STUDENT_ID}

export TMPDIR=/disk/scratch/${STUDENT_ID}/
export TMP=/disk/scratch/${STUDENT_ID}/
# Activate the relevant virtual environment:

source /home/${STUDENT_ID}/miniconda3/bin/activate proj

python ../gan.py --w256 --mn40s-pivot --batch-norm --skip-connections --dropout --l1 --loss-gan 0.01 --loss-w2 1.0 --loss-tv 0.0 --epochs 10 --gen-dataset
