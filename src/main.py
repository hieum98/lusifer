import copy
from dataclasses import asdict
import datetime
import os
import yaml
from pathlib import Path
from typing import Any, List
from functools import partial
import torch
from torch.distributed.fsdp.api import ShardingStrategy
import lightning as L
from lightning.fabric.strategies import FSDPStrategy, DDPStrategy
from lightning import seed_everything
from transformers import PreTrainedTokenizer, HfArgumentParser
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
from transformers.models.mt5.modeling_mt5 import MT5Block
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaLayer

from src.eval.eval import eval_mteb, eval_multilingual
from src.data_modules.rep_learning_datamodule import RepLearningDataModule, get_dataloaders as get_rep_learning_dataloaders
from src.data_modules.pretraining_datamodule import PretrainingDataModule, get_dataloaders as get_pretraining_dataloaders
from src.models.lusifer import Lusifer, WrappedLusifer
from src.models.utils import get_wrapping_policy, get_activation_checkpointing_policy
from src.trainer.supervised_trainer import SupervisedTrainer
from src.trainer.gradcache_trainer import GradCacheTrainer
from src.trainer.alignment_trainer import AlignmentTrainer
from src.args import DataArguments, ModelArguments, TrainingArguments
from src.trainer.utils import choose_logger, get_cosine_schedule_with_warmup, get_trainable_parameters, trainable_filter


backbone_to_layer_type = {
    'mistral': MistralDecoderLayer,
    't5': MT5Block,
    'xlm-r': XLMRobertaLayer,
}

def main(
        fabric: L.Fabric,
        train_data: RepLearningDataModule,
        data_args: DataArguments,
        model_args: ModelArguments,
        training_args: TrainingArguments,
        is_alignmnent: bool = False,
        is_cross_batch_loss: bool = False,
        ):
    fabric.seed_everything(training_args.seed)

    # Initialize model
    with fabric.rank_zero_first():
        model = Lusifer(
            universal_learner_name_or_path=model_args.universal_learner_name_or_path,
            encoder_name_or_path=model_args.encoder_name_or_path,
            universal_learner_backbone_type=model_args.universal_learner_backbone_type,
            encoder_backbone_type=model_args.encoder_backbone_type,
            is_freeze_universal_learner=model_args.is_freeze_universal_learner,
            is_freeze_encoder=True if is_alignmnent else False,
            connection_type=model_args.connection_type,
            num_added_tokens=model_args.num_added_tokens,
            encoder_lora_name=model_args.encoder_lora_name,
            universal_learner_lora_name=model_args.universal_learner_lora_name,
            loar_r=model_args.loar_r,
            lora_alpha=model_args.lora_alpha,
            dropout=model_args.dropout,
            attn_implementation=model_args.attn_implementation,
        )
    tokenizer = model.tokenizer
    encoder_tokenizer = model.encoder_tokenizer
    trainable_params, all_param, trainable_params_percentage, trainable_layers = get_trainable_parameters(model)
    filter_fn = partial(trainable_filter, trainable_layers=trainable_layers) if trainable_params_percentage < 100 else None
    fabric.print(f"Number of trainable parameters: {trainable_params/1e6:.2f}M")
    fabric.print(f"Total number of parameters: {all_param/1e6:.2f}M")
    fabric.print(f"Percentage of trainable parameters: {trainable_params_percentage:.2f}%")
    model = fabric.setup_module(model)
    fabric.print("Model after wrapping")
    fabric.print(model)

    # Prepare the dataloaders
    if is_alignmnent:
        train_dataloader = get_pretraining_dataloaders(
            fabric=fabric,
            data_module=train_data,
            universal_learner_tokenizer=tokenizer,
            lm_tokenizer=encoder_tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
        )
    else:
        train_dataloader = get_rep_learning_dataloaders(
            fabric=fabric,
            data_module=train_data,
            tokenizer=tokenizer,
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            is_cross_batch_loss=is_cross_batch_loss,
        )
    fabric.barrier()

    # Setup the optimizer and scheduler
    if is_alignmnent or (not is_cross_batch_loss):
        num_accumulation_steps = training_args.global_batch_size // (training_args.gc_chunk_size * fabric.world_size)
        step_per_epoch = len(train_dataloader) // num_accumulation_steps
    else:
        num_accumulation_steps = 1
        step_per_epoch = len(train_dataloader)
    
    lr_max_steps = min(training_args.max_steps, step_per_epoch * training_args.max_epochs)
    warmup_steps = min(training_args.warmpup_proportion * lr_max_steps, 500)
    lr = training_args.learning_rate
    min_lr = training_args.min_learning_rate
    min_reduce_rate = min_lr / lr

    fabric.print(f"Number of accumulation steps: {num_accumulation_steps}")
    fabric.print(f"Number of steps per epoch: {step_per_epoch}")
    fabric.print(f"Data size: {len(train_dataloader)}")
    fabric.print(f"Number of max steps: {lr_max_steps}")
    fabric.print(f"Number of warmup steps: {warmup_steps}")
    fabric.print(f"Initial learning rate: {lr}")
    fabric.print(f"Minimum learning rate: {min_lr}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=training_args.weight_decay,
        betas=(0.9, 0.999),
    )
    optimizer = fabric.setup_optimizers(optimizer)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=lr_max_steps,
        min_reduce_rate=min_reduce_rate,
    )

    # Load the checkpoint if needed
    state = {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "current_step": 0,
        "epoch_num": 0,
    }
    checkpoint_path = None
    if training_args.checkpoint_file is not None:
        checkpoint_path = training_args.checkpoint_file
        if os.path.exists(checkpoint_path):
            fabric.print(f"Load checkpoint from {checkpoint_path}")
            if training_args.only_load_model:
                print(f"Only load model from {checkpoint_path}")
                model_state = {'model': model}
                remaining_keys = fabric.load(checkpoint_path, model_state, strict=False)
                remaining_keys = remaining_keys.get("model", None)
                fabric.print(f"Following keys are not loaded: {remaining_keys}")
                model = model_state.pop("model")
            else:
                print(f"Load all states from {checkpoint_path}")
                state['model'] = model
                remaining_keys = fabric.load(checkpoint_path, state, strict=False)
                fabric.print(f"Following keys are not loaded: {remaining_keys}")
                model = state.pop("model")

    # Evaluate the initial model
    if fabric.global_rank == 0:
        model_hprams = model.hprams
        _model_revision = f"{training_args.model_revision}_initial"
        eval_model = WrappedLusifer(
            model_revision=_model_revision, 
            model_checkpoint=checkpoint_path,
            **model_hprams
            )
        mteb_results = eval_mteb(
            model=eval_model,
            output_folder=training_args.checkpoint_dir,
            batch_size=training_args.eval_batch_size,
            is_quick_run=True,
        )
        multilingual_results = eval_multilingual(
            model=eval_model,
            langs=['en', 'ru', 'vi', 'zh'],
            output_folder=training_args.checkpoint_dir,
            batch_size=training_args.eval_batch_size,
            is_quick_run=True,
        )
        results = {
            'Avg/mteb_quick_avg': mteb_results['avg']['all_tasks'],
            'Avg/multilingual_quick_avg': multilingual_results['avg']['all_tasks'],
        }
        fabric.log_dict(results, step=0)
        # Eval logic here
        fabric.print("Model evaluation finished")
        fabric.print(f"Mteb results : {mteb_results}")
        fabric.print(f"Multilingual results : {multilingual_results}")
        del eval_model

    # Initialize the trainer
    if is_alignmnent:
        fabric.print("Alignment training using AlignmentTrainer")
        trainer = AlignmentTrainer(
            fabric=fabric,
            num_accumulation_steps=num_accumulation_steps,
        )
    elif is_cross_batch_loss:
        fabric.print("Representation learning with cross-batch loss using GradCacheTrainer")
        trainer = GradCacheTrainer(
            fabric=fabric,
            loss_type=training_args.loss_type,
            temperature=training_args.temperature,
            is_distance=training_args.is_distance,
            use_miner=training_args.use_miner,
            chunk_size=training_args.gc_chunk_size,
        )
    else:
        fabric.print("Representation learning using SupervisedTrainer")
        trainer = SupervisedTrainer(
            fabric=fabric,
            num_accumulation_steps=num_accumulation_steps,
        )

    current_epoch_num = state.get("epoch_num", 0)
    fabric.print(f"Start training from epoch {current_epoch_num}")
    for epoch in range(current_epoch_num, training_args.max_epochs):
        if is_alignmnent:
            train_dataloader = get_pretraining_dataloaders(
                fabric=fabric,
                data_module=train_data,
                universal_learner_tokenizer=tokenizer,
                lm_tokenizer=encoder_tokenizer,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                epoch=epoch,
            )
        else:
            train_dataloader = get_rep_learning_dataloaders(
                fabric=fabric,
                data_module=train_data,
                tokenizer=tokenizer,
                data_args=data_args,
                model_args=model_args,
                training_args=training_args,
                epoch=epoch,
                is_cross_batch_loss=is_cross_batch_loss,
            )
        fabric.barrier()
        checkpoint_path = trainer.fit_epoch(
            model=model,
            train_loader=train_dataloader,
            state=state,
            lr_max_steps=lr_max_steps,
            grad_norm_clip=training_args.grad_norm_clip,
            log_interval=training_args.log_interval,
            checkpoint_iterval=training_args.checkpoint_interval,
            checkpoint_dir=training_args.checkpoint_dir,
            checkpoint_filter=filter_fn,
            model_revision=training_args.model_revision,
            eval_batch_size=training_args.eval_batch_size,
        )
        fabric.barrier()
        # Reload the model from the checkpoint 
        torch.cuda.empty_cache()
        state['model'] = model
        fabric.load(checkpoint_path, state, strict=False)
        model = state.pop("model")
        if state["current_step"] > lr_max_steps * num_accumulation_steps:
            break
    
    fabric.print("Training is finished")


def setup(data_args: DataArguments, model_args: ModelArguments, training_args: TrainingArguments, is_alignment: bool = False, run_name: str = None):
    seed_everything(training_args.seed)
    is_cross_batch_loss = None
    if is_alignment:
        print(f"Alignment training that using {PretrainingDataModule} data module") 
        train_data = PretrainingDataModule(
            is_reconstruct=data_args.is_reconstruct,
            is_query_positive_alignment=data_args.is_query_positive_alignment,
            num_workers=data_args.num_workers,
            seed=training_args.seed
        )
    else:
        print(f"Representation learning training that using {RepLearningDataModule} data module")
        train_data = RepLearningDataModule(
            langs=data_args.langs,
            use_retrieval_data_only=data_args.use_retrieval_data_only,
            num_workers=data_args.num_workers,
            seed=training_args.seed
        )
        is_cross_batch_loss = data_args.use_retrieval_data_only

    strategy = training_args.strategy
    if training_args.nodes > 1 or training_args.devices > 1:
        if training_args.strategy == 'fsdp':
            # Config sharding strategy
            if training_args.sharding_strategy == "full_shard":
                sharding_strategy = ShardingStrategy.FULL_SHARD
            elif training_args.sharding_strategy == "shard_grad_op":
                sharding_strategy = ShardingStrategy.SHARD_GRAD_OP
            elif training_args.sharding_strategy == "ddp":
                sharding_strategy = ShardingStrategy.NO_SHARD
            elif training_args.sharding_strategy == "hybrid_full_shard":
                sharding_strategy = ShardingStrategy.HYBRID_SHARD
            elif training_args.sharding_strategy == "hybrid_shard_grad_op":
                sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2
            else:
                raise ValueError("Invalid sharding strategy")

            backbone_layer_types = {backbone_to_layer_type[model_args.encoder_backbone_type], 
                                    backbone_to_layer_type[model_args.universal_learner_backbone_type]}
            wrapping_policy = get_wrapping_policy(backbone_layer_types)
            activation_checkpointing_policy = get_activation_checkpointing_policy(backbone_layer_types)
            
            strategy = FSDPStrategy(
                auto_wrap_policy=wrapping_policy,
                activation_checkpointing_policy=activation_checkpointing_policy if training_args.activation_checkpointing else None,
                sharding_strategy=sharding_strategy,
                limit_all_gathers=True, # See https://github.com/pytorch/pytorch/issues/91165
                state_dict_type="full",
                cpu_offload=training_args.use_cpu_offload,
                timeout=datetime.timedelta(hours=5), # makeing large timeout for model evaluation
            )
        elif training_args.strategy == 'ddp':
            strategy = DDPStrategy(
                find_unused_parameters=True, 
                timeout=datetime.timedelta(hours=5), # makeing large timeout for model evaluation
            )
    else:
        strategy = "auto"

    logger_dir = os.path.join(training_args.checkpoint_dir, f"logs_{training_args.logger_type}")
    os.makedirs(logger_dir, exist_ok=True)
    logger = choose_logger(
        logger_name=training_args.logger_type,
        out_dir=Path(logger_dir),
        project_name=training_args.logger_name,
        run_name=run_name,
        log_interval=training_args.log_interval,
    )

     # check whether gpu is support bf16
    if not torch.cuda.is_bf16_supported():
        training_args.precision = '16-mixed'

    fabric = L.Fabric(
        accelerator='gpu',
        strategy=strategy,
        devices=training_args.devices,
        num_nodes=training_args.nodes,
        precision=training_args.precision,
        loggers=logger,
    )
    fabric.launch(
        main,
        train_data=train_data,
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        is_alignmnent=is_alignment,
        is_cross_batch_loss=is_cross_batch_loss,
    )


if __name__ == "__main__":
    os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_DATASETS_TRUST_REMOTE_CODE']='1'
    
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    torch.set_float32_matmul_precision('high')
    parser.add_argument(
        "--config_file", type=str, required=True, help="Path to the yaml config file",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Run name"
    )
    parser.add_argument(
        "--model_revision", type=str, default=None, help="Model revision"
    )
    parser.add_argument(
        "--nodes", type=int, default=None, help="Number of nodes"
    )
    parser.add_argument(
        "--devices", type=int, default=None, help="Number of devices"
    )
    parser.add_argument(
        "--gc_chunk_size", type=int, default=None, help="Gradient cache chunk size"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=None, help="Learning rate"
    )
    parser.add_argument(
        "--min_learning_rate", type=float, default=None, help="Minimum learning rate"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None, help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--checkpoint_file", type=str, default=None, help="Checkpoint file to resume training"
    )
    parser.add_argument(
        "--only_load_model", action="store_true", help="Only load the model from the checkpoint"
    )

    args = parser.parse_args()
    config_file = args.config_file

    hf_parser = HfArgumentParser((DataArguments, ModelArguments, TrainingArguments))
    print(f"Loading yaml config {config_file}")
    data_args, model_args, training_args = hf_parser.parse_yaml_file(yaml_file=config_file)
    # Add read-only arguments
    if args.model_revision is not None:
        training_args.model_revision = args.model_revision
    training_args.nodes = args.nodes if args.nodes is not None else training_args.nodes
    training_args.devices = args.devices if args.devices is not None else training_args.devices
    training_args.gc_chunk_size = args.gc_chunk_size if args.gc_chunk_size is not None else training_args.gc_chunk_size
    training_args.learning_rate = args.learning_rate if args.learning_rate is not None else training_args.learning_rate
    training_args.min_learning_rate = args.min_learning_rate if args.min_learning_rate is not None else training_args.min_learning_rate
    training_args.checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else training_args.checkpoint_dir
    training_args.checkpoint_file = args.checkpoint_file if args.checkpoint_file is not None else training_args.checkpoint_file
    training_args.only_load_model = args.only_load_model

    config_file_path = Path(training_args.checkpoint_dir) / "config.yaml"
    config_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_file_path, "w") as f:
        yaml.dump(asdict(data_args), f)
        yaml.dump(asdict(model_args), f)
        yaml.dump(asdict(training_args), f)

    setup(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        is_alignment=training_args.is_alignment,
        run_name=args.run_name,
    )