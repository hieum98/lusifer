import os
from typing import List
import torch
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
import lightning as L
from transformers import PreTrainedTokenizer

from lusifer.args import DataArguments, ModelArguments, TrainingArguments
from lusifer.data_modules.constants import *
from lusifer.data_modules.pretraining_dataset import AlignmentDataset, PretrainingCollator
from lusifer.special_tokens import SPECIAL_TOKENS


class PretrainingDataModule(L.LightningDataModule):
    def __init__(
            self, 
            num_workers: int = 4,
            is_reconstruct: bool = True,
            is_query_positive_alignment: bool = False,
            mask_probability: float = 0.15,
            seed: int = 777
            ):
        super().__init__()
        assert is_reconstruct or is_query_positive_alignment, "At least one of is_reconstruct or is_query_positive_alignment must be True."
        if is_reconstruct and is_query_positive_alignment:
            self.data_names = PRETRAINING_RECONSTRUCT + PRETRAINING_PASSAGE2QUERY + PRETRAINING_QUERY2PASSAGE
        elif is_reconstruct:
            self.data_names = PRETRAINING_RECONSTRUCT
        elif is_query_positive_alignment:
            self.data_names = PRETRAINING_PASSAGE2QUERY + PRETRAINING_QUERY2PASSAGE
        
        self.data_names.sort()
        self.mask_probability = mask_probability
        self.num_workers = num_workers
        self.seed = seed

    def connect(
            self,
            universal_learner_tokenizer: PreTrainedTokenizer,
            lm_tokenizer: PreTrainedTokenizer,
            universal_learner_special_tokens_set: str,
            lm_special_tokens_set: str,
            world_size: int = 1,
            global_rank: int = 0,
            global_batch_size: int = 32,
            max_seq_length: int = 512,
            number_training_samples: int = 1_000_000,
            ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.universal_learner_tokenizer = universal_learner_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.universal_learner_special_tokens_set = SPECIAL_TOKENS[universal_learner_special_tokens_set]
        self.lm_special_tokens_set = SPECIAL_TOKENS[lm_special_tokens_set]
        self.global_batch_size = global_batch_size
        self.max_seq_length = max_seq_length
        self.number_training_samples = number_training_samples
        self.batch_size = self.global_batch_size // self.world_size
        if self.global_batch_size % self.world_size != 0:
            self.global_batch_size = self.batch_size * self.world_size
            print(f"Global batch size must be divisible by world size. Setting global batch size to {self.global_batch_size}")
        if self.batch_size <= 0:
            self.batch_size = 1
            self.global_batch_size = self.world_size
            print(f"Batch size must be greater than 0. i.e. world_size must be less than or equal to global_batch_size. Setting batch size to {self.batch_size}")

    def set_epoch(self, epoch: int) -> None:
        self.seed = self.seed + epoch

    def setup(self, stage: str='') -> None:
        train_datasets = []
        for data_name in self.data_names:
            ds = AlignmentDataset(
                data_name=data_name,
                number_training_samples=self.number_training_samples,
                seed=self.seed,
            )
            if len(ds) > 0:
                train_datasets.append(ds)
                if self.global_rank == 0:
                    print(f"Loaded {data_name} dataset with {len(ds)} samples.")
            else:
                print(f"Skipping {data_name} dataset as it has no samples.")
        assert len(train_datasets) > 0, f"No datasets loaded. Please check the data names: {self.data_names}"
        self.train_ds = ConcatDataset(train_datasets)

    def train_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
        collator = PretrainingCollator(
            universal_learner_tokenizer=self.universal_learner_tokenizer,
            lm_tokenizer=self.lm_tokenizer,
            universal_learner_special_tokens=self.universal_learner_special_tokens_set,
            lm_special_tokens=self.lm_special_tokens_set,
            max_seq_length=self.max_seq_length,
            mask_probability=self.mask_probability,
        )
        sampler = DistributedSampler(
            self.train_ds,
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=True,
            seed=self.seed,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collator,
        )


def get_dataloaders(
        fabric: L.Fabric, 
        data_module: PretrainingDataModule,
        universal_learner_tokenizer: PreTrainedTokenizer,
        lm_tokenizer: PreTrainedTokenizer,
        data_args: DataArguments, 
        model_args: ModelArguments,
        training_args: TrainingArguments,
        epoch: int = 0,
        ):
    data_module.connect(
        world_size=fabric.world_size,
        global_rank=fabric.global_rank,
        universal_learner_tokenizer=universal_learner_tokenizer,
        lm_tokenizer=lm_tokenizer,
        universal_learner_special_tokens_set=model_args.universal_learner_backbone_type,
        lm_special_tokens_set=model_args.encoder_backbone_type,
        global_batch_size=training_args.gc_chunk_size * fabric.world_size,
        max_seq_length=data_args.max_seq_length,
        number_training_samples=data_args.number_training_samples,
    )
    data_module.set_epoch(epoch)
    with fabric.rank_zero_first():
        data_module.setup()
        train_dataloader = data_module.train_dataloader()
        train_dataloader = fabric.setup_dataloaders(
            train_dataloader,
            use_distributed_sampler=False,
            move_to_device=True
        )
    return train_dataloader


if __name__=='__main__':
    from transformers import AutoTokenizer
    from lightning import seed_everything
    from tqdm import tqdm

    seed_everything(777)
    universal_learner_tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')
    lm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B')
    if lm_tokenizer.pad_token is None:
        lm_tokenizer.pad_token = SPECIAL_TOKENS['llama']['pad']
    if lm_tokenizer.mask_token is None:
        lm_tokenizer.mask_token = SPECIAL_TOKENS['llama']['mask']
    if universal_learner_tokenizer.pad_token is None:
        universal_learner_tokenizer.pad_token = universal_learner_tokenizer.eos_token
    dm = PretrainingDataModule(
        num_workers=0,
        seed=777,
    )
    dm.connect(
        universal_learner_tokenizer=universal_learner_tokenizer,
        lm_tokenizer=lm_tokenizer,
        universal_learner_special_tokens_set='xlm-r',
        lm_special_tokens_set='mistral',
        world_size=1,
        global_rank=0,
        global_batch_size=32,
        max_seq_length=512,
    )
    dm.setup()
    dl = dm.train_dataloader()
    for batch in tqdm(dl):
        breakpoint()




