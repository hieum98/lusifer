python -m lusifer.eval.eval \
    --model_name_or_path BAAI/bge-large-en-v1.5 \
    --output_folder bge-en \
    --batch_size 512 \
    --max_length 512 \
    --langs all