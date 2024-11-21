python -m src.eval.eval \
    --model_name_or_path sentence-transformers/gtr-t5-xxl \
    --output_folder /sensei-fs/users/chienn/hieu/lusifer/gtr \
    --batch_size 512 \
    --max_length 512 \
    --langs all \
    --num_gpus 1