from dataclasses import dataclass, field
from typing import List


@dataclass
class DataArguments:
    langs: List[str] = field(
        default_factory=lambda: ['en'],
        metadata={"help": "The languages to use for training."}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "The maximum sequence length."}
    )
    number_training_samples: int = field(
        default=1_000_000,
        metadata={"help": "The number of training samples per dataset."}
    )
    neg_per_sample: int = field(
        default=1,
        metadata={"help": "The number of negative samples per sample."}
    )
    pos_per_sample: int = field(
        default=1,
        metadata={"help": "The number of positive samples per sample."}
    )
    num_workers: int = field(
        default=0,
        metadata={"help": "Number of workers to use for data loading"}
    )


@dataclass
class ModelArguments:
    univeral_learner_name_or_path: str = field(
        default='google-t5/t5-large',
        metadata={"help": "The name or path of the universal learner model."}
    )
    encoder_name_or_path: str = field(
        default='mistralai/Mistral-7B-Instruct-v0.3',
        metadata={"help": "The name or path of the encoder model."}
    )
    univeral_learner_backbone_type: str = field(
        default='t5',
        metadata={"help": "The type of the universal learner backbone."}
    )
    encoder_backbone_type: str = field(
        default='mistral',
        metadata={"help": "The type of the encoder backbone."}
    )
    is_freeze_univeral_learner: bool = field(
        default=True,
        metadata={"help": "Whether to freeze the universal learner model or not."} 
    )
    encoder_lora_name: str = field(
        default=None,
        metadata={"help": "The name of the encoder LoRA layer."}
    )
    universal_learner_lora_name: str = field(
        default=None,
        metadata={"help": "The name of the universal learner LoRA layer."}
    )
    loar_r: int = field(
        default=16,
        metadata={"help": "LoRA r parameter."}
    )
    lora_alpha: int = field(
        default=32,
        metadata={"help": "LoRA alpha parameter."}
    )
    dropout: float = field(
        default=0.1,
        metadata={"help": "The dropout probability."}
    )
    attn_implementation: str = field(
        default='flash_attention_2',
        metadata={"help": "The attention implementation."}
    )


@dataclass
class TrainingArguments:
    seed: int = field(
        default=777,
        metadata={"help": "Seed for reproducibility"}
    )

    model_revision: str = field(
        default='dev.v0',
        metadata={"help": "Model revision"}
    )
    
    nodes: int = field(
        default=1,
        metadata={"help": "Number of nodes to use for training"}
    )
    devices: int = field(
        default=1,
        metadata={"help": "Number of devices per node to use for training"}
    )
    precision: str = field(
        default='bf16-true',
        metadata={"help": "Precision to use. Can be bf16-true/bf16-mixed/16-mixed/32"}
    )
    strategy: str = field(
        default='fsdp',
        metadata={"help": "Strategy to use. Currently only supports dpp and fsdp"}
    )
    use_cpu_offload: bool = field(
        default=False,
        metadata={"help": "Whether to use CPU offload or not"}
    )
    sharding_strategy: str = field(
        default='full_shard',
        metadata={"help": "Sharding strategy to use. Can be full_shard/shard_grad_op/ddp/hybrid_full_shard/hybrid_shard_grad_op"}
    )
    quantization: bool = field(
        default=False,
        metadata={"help": "Whether to use quantization pretrained model. Note that, this is not supported in FSDP multi-GPU training yet."}
    )
    activation_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether to use activation checkpointing or not"}
    )
    
    loss_type: str = field(
        default='NTXentLoss',
        metadata={"help": "The Contrastive Loss function to use. Can be 'NTXentLoss' or 'SupConLoss'"}
    )
    temperature: float = field(
        default=0.05,
        metadata={"help": "Temperature for Contrastive loss"}
    )
    use_miner: bool = field(
        default=True,
        metadata={"help": "Whether to use miner or not. The MultiSimilarityMiner will be used."}
    )
    is_distance: bool = field(
        default=True,
        metadata={"help": "Whether to use distance metric or not. If True, LpDistance will be used, otherwise CosineSimilarity."}
    )

    global_batch_size: int = field(
        default=32,
        metadata={"help": "The global batch size."}
    )
    gc_chunk_size: int = field(
        default=1,
        metadata={"help": "GradCache chunk size. If None, not use GradCache."}
    )
    eval_batch_size: int = field(
        default=32,
        metadata={"help": "Evaluation batch size"}
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Maximum number of epochs to train"}
    )
    max_steps: int = field(
        default=float("inf"),
        metadata={"help": "Maximum number of steps to train"}
    )
    learning_rate: float = field(
        default=1e-4,
        metadata={"help": "Learning rate"}
    )
    min_learning_rate: float = field(
        default=0.0,
        metadata={"help": "Minimum learning rate"}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "Weight decay to apply."},
        )
    warmpup_proportion: float = field(
        default=0.1,
        metadata={"help": "Proportion of training steps to perform linear learning rate warmup for. E.g., 0.1 = 10% of training."}
    )
    grad_norm_clip: float = field(
        default=None,
        metadata={"help": "Gradient norm clipping value"}
    )
    
    checkpoint_dir: str = field(
        default=None,
        metadata={"help": "Directory to save checkpoints"}
    )
    checkpoint_file: str = field(
        default=None,
        metadata={"help": "File to save checkpoints"}
    )
    checkpoint_interval: int = field(
        default=1000,
        metadata={"help": "Interval to save the checkpoint"}
    )
    logger_type: str = field(
        default='wandb',
        metadata={"help": "Name of the logger to use. Can be wandb/tensorboard"}
    )
    logger_name: str = field(
        default='default',
        metadata={"help": "Name of the logger"}
    )
    log_interval: int = field(
        default=1,
        metadata={"help": "Interval to log the training progress"}
    )
