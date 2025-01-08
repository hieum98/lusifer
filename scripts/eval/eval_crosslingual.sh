#!/bin/bash

#SBATCH --nodes=1              # This needs to match Fabric(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb|h100|gpu-40gb
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5
#SBATCH --job-name=lusifer-eval
#SBATCH --partition=preempt,gpulong,cisds
#SBATCH --account=uonlp
#SBATCH --output=/home/hieum/uonlp/lusifer/lusifer-eval-%j.log
#SBATCH --error=/home/hieum/uonlp/lusifer/lusifer-eval-%j.log
#SBATCH --array=0-9

# Activate conda environment
source /home/hieum/.bashrc
conda activate lusifer
cd /home/hieum/uonlp/lusifer


DATA=(
'IndicCrosslingualSTS'
'MLQARetrieval'
'BelebeleRetrieval'
'STS17'
'mFollowIRCrossLingualInstructionRetrieval'
'XPQARetrieval'
'STS22.v2'
'CrossLingualSemanticDiscriminationWMT21'
'CrossLingualSemanticDiscriminationWMT19'
)

MODEL_NAMES=(
# princeton-nlp/sup-simcse-roberta-large
# intfloat/multilingual-e5-large
# facebook/contriever
# sentence-transformers/gtr-t5-xxl
# sentence-transformers/sentence-t5-xxl
# thenlper/gte-large
# BAAI/bge-large-en-v1.5
# intfloat/e5-large
# intfloat/e5-mistral-7b-instruct
sentence-transformers/paraphrase-multilingual-mpnet-base-v2
jinaai/jina-embeddings-v3
Alibaba-NLP/gte-multilingual-base
BAAI/bge-m3
)

OUTPUT_FOLDERS=(
# experiments/simcse
# experiments/me5-large
# experiments/contriever
# experiments/gtr
# experiments/st5
# experiments/gte
# experiments/bge-en
# experiments/e5-large
# experiments/e5-mistral
experiments/mmpnet-v2
experiments/jina-v3
experiments/mgte
experiments/bge-m3
)

export HF_HOME="/home/hieum/uonlp/hf_data_mteb" 

MODEL_NAME=${MODEL_NAMES[$SLURM_ARRAY_TASK_ID]}
OUTPUT_FOLDER=${OUTPUT_FOLDERS[$SLURM_ARRAY_TASK_ID]}

# Check if MODEL_NAME is empty
if [ -z "$MODEL_NAME" ]; then
    echo "MODEL_NAME is empty"
    rm -rf /home/hieum/uonlp/lusifer/lusifer-eval-$SLURM_JOB_ID.log
fi

# MODEL_NAME=output/v0.1/xlm-mistral-v0.3.v0/config.yaml
# MODEL_CHECKPOINT=output/v0.1/xlm-mistral-v0.3.v0/best_multi.ckpt
# OUTPUT_FOLDER=experiments/lusifer.mistral-v0.3
# DATANAME=${DATA[$SLURM_ARRAY_TASK_ID]}

echo "MODEL_NAME: $MODEL_NAME"
echo "OUTPUT_FOLDER: $OUTPUT_FOLDER"

python -m lusifer.eval.eval \
    --model_name_or_path $MODEL_NAME \
    --output_folder $OUTPUT_FOLDER \
    --batch_size 512 \
    --max_length 512 \
    --eval_crosslingual

# Check if the job is done
if [ $? -eq 0 ]; then
    echo "Job is done"
    rm -rf /home/hieum/uonlp/lusifer/lusifer-eval-$SLURM_JOB_ID.log
else
    echo "Job failed"
fi



