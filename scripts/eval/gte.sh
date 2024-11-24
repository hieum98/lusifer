python -m src.eval.eval \
    --model_name_or_path thenlper/gte-large \
    --output_folder gte \
    --batch_size 512 \
    --max_length 512 \
    --langs all