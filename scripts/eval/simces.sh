python -m lusifer.eval.eval \
    --model_name_or_path princeton-nlp/sup-simcse-roberta-large \
    --output_folder simcse \
    --batch_size 512 \
    --max_length 512 \
    --langs all
