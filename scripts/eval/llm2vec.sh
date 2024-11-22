
pip install llm2vec

python -m src.eval.eval \
    --model_name_or_path McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
    --output_folder /sensei-fs/users/chienn/hieu/lusifer/llm2vec-llama \
    --batch_size 512 \
    --max_length 512 \
    --langs all