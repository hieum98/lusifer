import random
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
import datasets

from src.data_modules.constants import DATA, PRETRAINING_DATASETS, PRETRAINING_PAIR_DATASETS


class PretrainingDataset(Dataset):
    def __init__(
            self,
            data_name: str,
            number_training_samples: int=1_000_000,
            seed: int=777,
            ):
        super().__init__()

        data_path = DATA[data_name]['data_path']
        self.data_name = data_name
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.instruction = DATA[data_name]['instruction']

        try:
            data = datasets.load_dataset(data_name, split='train')
        except:
            data = datasets.load_dataset('json', data_files=data_path, split='train')
        if len(data) > number_training_samples:
            data = data.train_test_split(train_size=number_training_samples, seed=seed, shuffle=True)['train']
        self.data = data
    
    def __len__(self):
        return len(self.data)


class AlignmentDataset(PretrainingDataset):
    def only_alignment(self, idx):
        example = self.data[idx]
        if self.data_name in PRETRAINING_DATASETS:
            text = example['query']
        else:
            pos = self.rng.choice(example['positive'])
            text = f"{example['query']}. {pos}"
        return {'query': text, 'instruction': ""} # instruction is empty string for reconstruction case
    
    def query_positive_alignment(self, idx):
        example = self.data[idx]
        assert self.data_name in PRETRAINING_PAIR_DATASETS, f"Data name {self.data_name} not in {PRETRAINING_PAIR_DATASETS}"
        text = example['query']
        pos = self.rng.choice(example['positive'])
        return {'query': text, 'answer': pos, 'instruction': self.instruction}

    def __getitem__(self, idx):
        if self.data_name in PRETRAINING_PAIR_DATASETS and self.rng.random() < 0.5:
            return self.query_positive_alignment(idx)
        else:
            return self.only_alignment(idx)


class PretrainingCollator:
    def __init__(
            self, 
            universal_learner_tokenizer: PreTrainedTokenizer,
            lm_tokenizer: PreTrainedTokenizer,
            universal_learner_special_tokens: Dict[str, str],
            lm_special_tokens: Dict[str, str],
            max_seq_length: int=512,
            label_pad_token_id: int=-100,
            ):
        self.universal_learner_tokenizer = universal_learner_tokenizer
        self.lm_tokenizer = lm_tokenizer
        universal_learner_bos = universal_learner_special_tokens.get("bos", "")
        universal_learner_eos = universal_learner_special_tokens.get("eos", "")
        self.universal_learner_format = universal_learner_bos + "{instruction}." + "\n{example}"  + universal_learner_eos
        self.lm_format = lm_special_tokens.get("bos", "") + "{text}" + lm_special_tokens.get("eos", "")

        self.max_seq_length = max_seq_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, examples):
        universal_learner_input_texts = []
        lm_input_texts = []
        for example in examples:
            query = example['query']
            instruction = example['instruction']
            universal_learner_input_texts.append(self.universal_learner_format.format(instruction=instruction, example=query))
            # if 'answer' in example is the query_positive_alignment case else is the reconstruction case
            lm_input_text = "{instruction}.\n{example}".format(instruction=instruction, example=query) if 'answer' not in example else example['answer']
            lm_input_texts.append(self.lm_format.format(text=lm_input_text))
        
        universal_learner_encodings = self.universal_learner_tokenizer(
            universal_learner_input_texts, 
            padding='longest',
            truncation=True, 
            max_length=self.max_seq_length, 
            add_special_tokens=False, # special tokens are already added
            return_tensors='pt'
            )
        lm_encodings = self.lm_tokenizer(
            lm_input_texts, 
            padding='longest',
            truncation=True, 
            max_length=self.max_seq_length, 
            add_special_tokens=False, # special tokens are already added
            return_tensors='pt'
            )

        return {
            'universal_learner_input_ids': universal_learner_encodings['input_ids'], # (b, l)
            'universal_learner_attention_mask': universal_learner_encodings['attention_mask'], # (b, l)
            'lm_input_ids': lm_encodings['input_ids'], # (b, l)
            'lm_attention_mask': lm_encodings['attention_mask'], # (b, l)
        }
    

