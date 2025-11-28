#!/bin/bash

#SBATCH --job-name=accuracy
#SBATCH -N 1 ## number of nodes
#SBATCH -n 1 ## number of tasks
#SBATCH -p gpu-a100-small # Queue (partition) name
#SBATCH -o ./slurm_logs/%x.o%j       # Name of stdout output file, %x for job name, %j for job id
#SBATCH -e ./slurm_logs/%x.e%j       # Name of stderr error file
#SBATCH -t 05:00:00
#SBATCH -A IFML
#SBATCH --mail-type=all
#SBATCH --mail-user=trungnguyen@utexas.edu

## record slurm script
cp $0 slurm_logs/job-$SLURM_JOB_ID-script.sh

nodes=($(scontrol show hostnames $SLURM_JOB_NODELIST))
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=DEBUG

export NCCL_DEBUG=INFO

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=1800

export NCCL_P2P_DISABLE=1

export TORCH_NCCL_BLOCKING_WAIT=1

export CUDA_LAUNCH_BLOCKING=1

export HF_HOME=$WORK/../ls6/huggingface_cache
export TMPDIR=/tmp/trung6/

export PYTHONPATH=$PYTHONPATH:$SCRATCH/artifact-eval/

# source $WORK/../ls6/miniconda3/bin/activate art_vllm_0.6
# export JAVA_HOME=/work/08955/trung6/ls6/miniconda3/envs/art_vllm_0.6
# export PATH=$JAVA_HOME/bin:$PATH
# export PYSPARK_PYTHON=/work/08955/trung6/ls6/miniconda3/envs/art_vllm_0.6/bin/python
# export PYSPARK_DRIVER_PYTHON=/work/08955/trung6/ls6/miniconda3/envs/art_vllm_0.6/bin/python

source $WORK/../ls6/miniconda3/bin/activate art_vllm_0.10
export JAVA_HOME=/work/08955/trung6/ls6/miniconda3/envs/art_vllm_0.10
export PATH=$JAVA_HOME/bin:$PATH
export PYSPARK_PYTHON=/work/08955/trung6/ls6/miniconda3/envs/art_vllm_0.10/bin/python
export PYSPARK_DRIVER_PYTHON=/work/08955/trung6/ls6/miniconda3/envs/art_vllm_0.10/bin/python

cd run/accuracy

# python llama_accuracy.py --model llama2-7b-chat-hf --dataset beer # --reordered

# python llama_accuracy.py --model llama2-13b-chat-hf --dataset beer # --reordered

# # python llama_accuracy.py --model llama3-8b --dataset movies --reordered

# python llama_accuracy.py --model gemma-2-2b-it --dataset beer # --reordered

# python llama_accuracy.py --model Mistral-7B-Instruct --dataset beer #--reordered

# python llama_accuracy_fever.py --model gemma-2-2b-it

# python llama_accuracy_fever.py --model llama3-8b

# python llama_accuracy_fever.py --model llama2-7b-chat-hf

# python llama_accuracy_fever.py --model llama2-13b-chat-hf

# python llama_accuracy_fever.py --model Mistral-7B-Instruct

python llama_accuracy_fever.py --model gemma-3-1b-it

python llama_accuracy_fever.py --model gemma-3-4b-it

python llama_accuracy_fever.py --model gemma-3-270m-it
