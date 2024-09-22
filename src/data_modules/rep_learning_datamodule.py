import os
import random
from typing import Dict, List, Sized
import torch
from torch.utils.data import DataLoader, Sampler
import lightning as L
from datasets import load_dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.args import DataArguments, ModelArguments, TrainingArguments
from src.data_modules.constants import *
from src.data_modules.rep_learning_dataset import RepLearningDataset, ConcatRepLearningDataset, RepLearningCollator
from src.special_tokens import SPECIAL_TOKENS


class ConcatedDataSampler(Sampler):
    """
    A sampler for ConcatDataset that gurantees that each batch will comes from same dataset.
    """ 
    def __init__(
            self,
            each_data_sizes: List[int],
            global_batch_size: int,
            shuffle: bool = True,
            num_replicas: int = 1,
            rank: int = 0,
            seed: int = 777,
            drop_last: bool = False,
            ):
        """
        :param each_data_sizes: list of sizes of each dataset
        :param global_batch_size: global batch size
        :param shuffle: whether to shuffle the indices
        :param num_replicas: number of replicas i.e. number of gpus
        :param rank: rank of the current gpu
        :param seed: seed for random number generator
        :param drop_last: whether to drop the last batch if it is incomplete
        """
        self.each_data_sizes = each_data_sizes
        self.batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.indices = self.set_indices()
        self.num_samples = len(self.indices) // self.num_replicas

    def set_indices(self):
        """
        Set the indices for the sampler based on the each_data_sizes and global_batch_size to guarantee that each batch comes from the same dataset.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            indices = [torch.randperm(n, generator=g).tolist() for n in self.each_data_sizes]
        else:
            indices = [list(range(n)) for n in self.each_data_sizes]

        # increase the indices by the offset
        for i in range(len(self.each_data_sizes)):
            indices[i] = [idx + sum(self.each_data_sizes[:i]) for idx in indices[i]]
        batched_indices = []
        for data_indices in indices:
            _batched_indices = list(torch.split(torch.tensor(data_indices), self.batch_size))
            batched_indices.append(_batched_indices)
        
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in batched_indices:
            if len(b[-1]) < self.batch_size:
                incomplete_indices.append(b.pop())
        
        if self.drop_last is False and len(incomplete_indices) != 0:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=g).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(incomplete_indices, self.batch_size))
            if len(mixed_batches[-1]) < self.batch_size:
                mixed_batches.pop()
            batched_indices = sum(batched_indices, []) + mixed_batches
        else:
            batched_indices = sum(batched_indices, [])

        if self.shuffle:
            # Shuffle the batches 
            order = torch.randperm(len(batched_indices), generator=g).tolist()
        else:
            order = list(range(len(batched_indices)))
                         
        indices = []
        for batch_idx in order:
            indices.extend([int(i) for i in batched_indices[batch_idx]])
        return indices

    def __iter__(self):
        # subsample
        indices = self.indices[self.rank:len(self.indices):self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class ClusteringDataSampler(Sampler):
    """
    A sampler for clustering that gurantees that each batch will comes from same dataset and same cluster.
    """ 
    def __init__(
            self,
            each_data_sizes: List[int],
            global_batch_size: int,
            cluster_info: List[Dict[str, List[int]]],
            shuffle: bool = True,
            num_replicas: int = 1,
            rank: int = 0,
            seed: int = 777,
            drop_last: bool = False,
    ) -> None:
        self.cluster_info = cluster_info
        self.each_data_sizes = each_data_sizes
        self.batch_size = global_batch_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.indices = self.set_indices()
        self.num_samples = len(self.indices) // self.num_replicas

    def set_indices(self):
        """
        Set the indices for the sampler based on the each_data_sizes and global_batch_size to guarantee that each batch comes from the same dataset.
        """
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        rnd = random.Random(self.seed + self.epoch)
        assert len(self.cluster_info) == len(self.each_data_sizes), 'Number of datasets should be equal'
        indices = []
        for ds_cluster in self.cluster_info:
            _indices = []
            for _, in_cluster_ids in ds_cluster.items():
                in_cluster_ids = list(in_cluster_ids)
                if self.shuffle:
                    rnd.shuffle(in_cluster_ids)
                _indices.extend(in_cluster_ids)
            indices.append(_indices)

        # increase the indices by the offset
        assert len(indices) == len(self.each_data_sizes), 'Number of datasets should be equal'
        for i in range(len(self.each_data_sizes)):
            assert len(indices[i]) == self.each_data_sizes[i], 'Number of samples should be equal'
            indices[i] = [idx + sum(self.each_data_sizes[:i]) for idx in indices[i]]
        batched_indices = []
        for data_indices in indices:
            _batched_indices = list(torch.split(torch.tensor(data_indices), self.batch_size))
            batched_indices.append(_batched_indices)
        
        # Create separate batches from the remaining samples
        incomplete_indices = []
        for b in batched_indices:
            if len(b[-1]) < self.batch_size:
                incomplete_indices.append(b.pop())
        
        if self.drop_last is False and len(incomplete_indices) != 0:
            # Randomly permute the incomplete indices
            order = torch.randperm(len(incomplete_indices), generator=g).tolist()
            incomplete_indices = torch.cat([incomplete_indices[i] for i in order])
            # Then split again into groups of four & drop the last one if it is incomplete
            mixed_batches = list(torch.split(incomplete_indices, self.batch_size))
            if len(mixed_batches[-1]) < self.batch_size:
                mixed_batches.pop()
            batched_indices = sum(batched_indices, []) + mixed_batches
        else:
            batched_indices = sum(batched_indices, [])

        if self.shuffle:
            # Shuffle the batches 
            order = torch.randperm(len(batched_indices), generator=g).tolist()
        else:
            order = list(range(len(batched_indices)))
                         
        indices = []
        for batch_idx in order:
            indices.extend([int(i) for i in batched_indices[batch_idx]])
        return indices

    def __iter__(self):
        # subsample
        indices = self.indices[self.rank:len(self.indices):self.num_replicas]
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class RepLearningDataModule(L.LightningDataModule):
    def __init__(
            self, 
            langs: List[str], 
            use_retrieval_data_only: bool = False,
            num_workers: int = 4,
            seed: int = 777
            ):
        super().__init__()
        lang_to_data = {
            'en': EN_CROSS_BATCH if use_retrieval_data_only else EN_CROSS_BATCH + EN_NON_CROSS_BATCH,
            'ar': AR,
            'bn': BN,
            'de': DE,
            'es': ES,
            'fa': FA,
            'fi': FI,
            'fr': FR,
            'hi': HI,
            'id': ID,
            # 'ja': JA,
            'ko': KO,
            'ru': RU,
            'sw': SW,
            'te': TE,
            'th': TH,
            'vi': VI,
            'yo': YO,
            'zh': ZH,
        }

        self.data_names = []
        if langs == ['all']:
            for l in lang_to_data.keys():
                self.data_names.extend(lang_to_data[l])
        else:
            for l in langs:
                self.data_names.extend(lang_to_data[l])
        self.data_names = list(set(self.data_names))
        self.data_names.sort()
        print(f"Data names: {self.data_names}")
        self.num_workers = num_workers
        self.seed = seed
    
    def connect(
            self,
            world_size: int = 1,
            global_rank: int = 0,
            tokenizer: PreTrainedTokenizer = None, 
            special_tokens_set: str = 't5',
            global_batch_size: int = 32,
            max_seq_length: int = 512,
            number_training_samples: int = 1_000_000,
            neg_per_sample: int = 1,
            pos_per_sample: int = 1,
            ) -> None:
        self.world_size = world_size
        self.global_rank = global_rank
        self.tokenizer = tokenizer
        self.special_tokens_set = SPECIAL_TOKENS[special_tokens_set]
        self.global_batch_size = global_batch_size
        self.max_seq_length = max_seq_length
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
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

    def prepare_data(self) -> None:
        for data_name in self.data_names:
            print(f"Loading {data_name} dataset.")
            # Download the dataset if not already downloaded
            load_dataset(data_name)

    def setup(self, stage: str='') -> None:
        train_datasets = []
        for data_name in self.data_names:
            ds = RepLearningDataset(
                data_name=data_name,
                number_training_samples=self.number_training_samples,
                neg_per_sample=self.neg_per_sample,
                pos_per_sample=self.pos_per_sample,
                seed=self.seed,
            )
            if len(ds) > 0:
                train_datasets.append(ds)
                if self.global_rank == 0:
                    print(f"Loaded {data_name} dataset with {len(ds)} samples.")
            else:
                print(f"Skipping {data_name} dataset as it has no samples.")
        assert len(train_datasets) > 0, f"No datasets loaded. Please check the data names: {self.data_names}"
        self.train_ds = ConcatRepLearningDataset(train_datasets)
    
    def train_dataloader(self) -> DataLoader:
        max_num_worker_suggest = 1
        try:
            max_num_worker_suggest = len(os.sched_getaffinity(0))
        except Exception:
            pass
        num_workers = min(self.num_workers, max_num_worker_suggest)
        collator = RepLearningCollator(
            tokenizer=self.tokenizer,
            special_tokens=self.special_tokens_set,
            max_seq_length=self.max_seq_length,
            label_pad_token_id=-100
        )
        each_data_sizes = [len(dataset) for dataset in self.train_ds.datasets]
        cluster_infor = [dataset.cluster for dataset in self.train_ds.datasets]
        sampler = ClusteringDataSampler(
            each_data_sizes=each_data_sizes,
            cluster_info=cluster_infor,
            global_batch_size=self.global_batch_size,
            shuffle=True,
            num_replicas=self.world_size,
            rank=self.global_rank,
            seed=self.seed,
            drop_last=False,
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
        data_module: RepLearningDataModule,
        tokenizer: PreTrainedTokenizer,
        data_args: DataArguments, 
        model_args: ModelArguments,
        training_args: TrainingArguments,
        epoch: int = 0,
        is_cross_batch_loss=True
        ):
    print(f"Creating dataloaders for epoch {epoch} with cross_batch_loss={is_cross_batch_loss}")
    batch_size = training_args.global_batch_size if is_cross_batch_loss else training_args.gc_chunk_size * fabric.world_size
    print(f"Batch size: {batch_size}")
    data_module.connect(
        world_size=fabric.world_size,
        global_rank=fabric.global_rank,
        tokenizer=tokenizer, 
        special_tokens_set=model_args.universal_learner_backbone_type,
        global_batch_size=training_args.global_batch_size if is_cross_batch_loss else training_args.gc_chunk_size * fabric.world_size,
        max_seq_length=data_args.max_seq_length,
        number_training_samples=data_args.number_training_samples,
        neg_per_sample=data_args.neg_per_sample,
        pos_per_sample=data_args.pos_per_sample,
    )
    if fabric.global_rank == 0:
        data_module.prepare_data()
    fabric.barrier()
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

    seed_everything(777)
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')
    dm = RepLearningDataModule(langs=['en'], num_workers=0, seed=777, use_retrieval_data_only=False)
    dm.connect(
        world_size=4,
        global_rank=0,
        tokenizer=tokenizer,
        special_tokens_set='xlm-r',
        global_batch_size=64,
        max_seq_length=512,
        number_training_samples=1_000_000,
        neg_per_sample=32000,
        pos_per_sample=3000,
    )
    dm.setup()
    dl = dm.train_dataloader()
    for batch in tqdm(dl): 
        pass


