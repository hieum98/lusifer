pip install gritlm

python -m src.eval.eval \
    --model_name_or_path GritLM/GritLM-7B \
    --output_folder /sensei-fs/users/chienn/hieu/lusifer/gritlm \
    --batch_size 64 \
    --max_length 512 \
    --langs all

