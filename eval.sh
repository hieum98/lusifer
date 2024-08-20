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
export HF_HOME="~/hf_cache"
export HUGGINGFACE_TOKEN="your_token"
export WANDB_API_KEY=""
wandb login $WANDB_API_KEY
huggingface-cli login --token $HUGGINGFACE_TOKEN


python -m src.eval.eval --model_name_or_path intfloat/multilingual-e5-large --output_folder $home/lusifer/me5 --batch_size 256

python -m src.eval.eval --model_name_or_path BAAI/bge-multilingual-gemma2 --output_folder $home/lusifer/mbge-gemma2 --batch_size 128

python -m src.eval.eval --model_name_or_path Alibaba-NLP/gte-multilingual-base --output_folder $home/lusifer/mgte --batch_size 256

python -m src.eval.eval --model_name_or_path BAAI/bge-m3 --output_folder $home/lusifer/bge-m3 --batch_size 256

python -m src.eval.eval --model_name_or_path izhx/udever-bloom-7b1 --output_folder $home/lusifer/udever-bloom-7B --batch_size 8

python -m src.eval.eval --model_name_or_path pkshatech/m-ST5 --output_folder $home/lusifer/mst5 --batch_size 128

python -m src.eval.eval --model_name_or_path sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --output_folder $home/lusifer/mmpnet-v2 --batch_size 256

python -m src.eval.eval --model_name_or_path yaushian/mSimCSE --output_folder $home/lusifer/msimcse --batch_size 256
