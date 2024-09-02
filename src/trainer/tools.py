import os
import torch
import torch.nn as nn
from transformers import HfArgumentParser

from src.args import DataArguments, ModelArguments, TrainingArguments
from src.models.lusifer import Lusifer


def merge_lora(
        config_path: str,
        checkpoint_path: str,
        merge_universal_learner_lora: bool = False,
        merge_lm_lora: bool = False,
        ):
    
    assert os.path.exists(config_path), f"Config file not found at {config_path}"
    assert os.path.exists(checkpoint_path), f"Checkpoint file not found at {checkpoint_path}"
    
    hf_parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    print(f"Loading yaml config {config_path}")
    model_args: ModelArguments = None

    data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_path)

    model = Lusifer(
            universal_learner_name_or_path=model_args.universal_learner_name_or_path,
            encoder_name_or_path=model_args.encoder_name_or_path,
            universal_learner_backbone_type=model_args.universal_learner_backbone_type,
            encoder_backbone_type=model_args.encoder_backbone_type,
            is_freeze_universal_learner=model_args.is_freeze_universal_learner,
            connection_type=model_args.connection_type,
            num_added_tokens=model_args.num_added_tokens,
            encoder_lora_name=model_args.encoder_lora_name,
            universal_learner_lora_name=model_args.universal_learner_lora_name,
            loar_r=model_args.loar_r,
            lora_alpha=model_args.lora_alpha,
            dropout=model_args.dropout,
            attn_implementation=model_args.attn_implementation,
        )
    print("Model created successfully")
    print(model)
    
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model.load_state_dict(state_dict['model'], strict=False)

    if merge_universal_learner_lora:
        if model_args.universal_learner_lora_name is None:
            raise ValueError("universal_learner_lora_name is None, thus cannot merge universal learner LoRA")
        model.universal_learner = model.universal_learner.merge_and_unload(progressbar=True)
    if merge_lm_lora:
        if model_args.encoder_lora_name is None:
            raise ValueError("encoder_lora_name is None, thus cannot merge LM LoRA")
        model.encoder = model.encoder.merge_and_unload(progressbar=True)

    return model


if __name__ == '__main__':
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--checkpoint_path", type=str, required=True, help="Path to the checkpoint file"
    )
    parser.add_argument(
        "--merge_universal_learner_lora", action='store_true', help="Whether to merge universal learner LoRA"
    )
    parser.add_argument(
        "--merge_lm_lora", action='store_true', help="Whether to merge LM LoRA"
    )
    parser.add_argument(
        "--output_path", type=str, default='merged_model.pth', help="Path to the output file"
    )

    args = parser.parse_args()
    model = merge_lora(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        merge_universal_learner_lora=args.merge_universal_learner_lora,
        merge_lm_lora=args.merge_lm_lora,
    )
    print("Model merged successfully")
    print(model)

    # Save the model using torch
    # Create the output directory if it does not exist
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    # Save the universal learner model
    if args.merge_universal_learner_lora:
        model_to_save = model.universal_learner
        save_dir = Path(args.output_path).parent / 'universal_learner'
        model_to_save.save_pretrained(save_dir)
        # Save the universal learner tokenizer
        universal_learner_tokenizer = model.tokenizer
        universal_learner_tokenizer.save_pretrained(save_dir)
        # Remove the universal learner from the model
        model.universal_learner = None
    if args.merge_lm_lora:
        model_to_save = model.encoder
        save_dir = Path(args.output_path).parent / 'encoder'
        model_to_save.save_pretrained(save_dir)
        # Save the encoder tokenizer
        encoder_tokenizer = model.encoder_tokenizer
        encoder_tokenizer.save_pretrained(save_dir)
        # Remove the encoder from the model
        model.encoder = None
    torch.save({'model': model.state_dict()}, args.output_path)






