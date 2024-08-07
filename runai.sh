export home=/sensei-fs/users/chienn/hieu

echo "Installing environment"
cp $home/Miniconda3-latest-Linux-x86_64.sh ~/
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

echo "Getting the code"
cd ~/
git clone https://github.com/hieum98/lusifer.git
cd lusifer
echo "Setting up the environment"
conda env create -f environment.yaml
conda activate lusifer
pip install -r requirements.txt
# Install flash-attn-2 
pip install ninja
pip install flash-attn --no-build-isolation

echo "Setting up the local variables"
export HUGGINGFACE_TOKEN="your_token"
export WANDB_API_KEY=""
wandb login $WANDB_API_KEY
huggingface-cli login --token $HUGGINGFACE_TOKEN

python -m src.main \
    --config_file scripts/configs/t5-mistral.yaml \
    --model_revision t5-mistral.v0.1 \
    --nodes 1 \
    --devices 8 \
    --gc_chunk_size 4 \
    --learning_rate 2e-4 \
    --min_learning_rate 5e-5 \
    --checkpoint_dir $home/lusifer/checkpoints/t5-mistral 

echo "Done"

