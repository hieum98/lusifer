python -m lusifer.fer.eval.eval \
    --model_name_or_path sentence-transformers/sentence-t5-xxl \
    --output_folder /sensei-fs/users/chienn/hieu/lusifer/st5 \
    --batch_size 512 \
    --max_length 512 \
    --langs all \
    --num_gpus 1
