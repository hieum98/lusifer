import random
from typing import Dict
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding
import datasets

from src.data_modules.constants import DATA, PRETRAINING_RECONSTRUCT, PRETRAINING_PASSAGE2QUERY, PRETRAINING_QUERY2PASSAGE


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
        if self.data_name in PRETRAINING_RECONSTRUCT:
            text = example['query']
        else:
            pos = self.rng.choice(example['positive'])
            text = f"{example['query']}. {pos}"
        return {'query': text, 'instruction': "Please reconstruct the following text."} # instruction is empty string for reconstruction case
    
    def query_positive_alignment(self, idx):
        example = self.data[idx]
        assert self.data_name in PRETRAINING_PASSAGE2QUERY or self.data_name in PRETRAINING_QUERY2PASSAGE, f"Data name {self.data_name} is not in {PRETRAINING_PASSAGE2QUERY} or {PRETRAINING_QUERY2PASSAGE}"
        if self.data_name in PRETRAINING_PASSAGE2QUERY:
            query = example['query']
            pos = self.rng.choice(example['positive'])
            if self.rng.random() < 0.3:
                return {'query': query, 'answer': pos, 'instruction': DATA[self.data_name]['instruction']}
            elif self.rng.random() < 0.4:
                return {'query': pos, 'instruction': "Please reconstruct the following text."}
            else:
                return {'query': pos, 'answer': query, 'instruction': "Please write a question based on this passage."}
        else:
            query = example['query']
            pos = self.rng.choice(example['positive'])
            if self.rng.random() < 0.1:
                return {'query': query, 'instruction': "Please reconstruct the following text."}
            else:
                return {'query': query, 'answer': pos, 'instruction': DATA[self.data_name]['instruction']}
                

    def __getitem__(self, idx):
        if self.data_name not in PRETRAINING_RECONSTRUCT:
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
            mask_probability: float=0.15,
            ):
        self.universal_learner_tokenizer = universal_learner_tokenizer
        self.lm_tokenizer = lm_tokenizer
        # Check if mask token is in the tokenizer else add it as a special token
        if self.lm_tokenizer.mask_token is None:
            self.lm_tokenizer.mask_token = self.lm_tokenizer.unk_token
            self.lm_tokenizer.mask_token_id = self.lm_tokenizer.unk_token_id

        universal_learner_bos = universal_learner_special_tokens.get("bos", "")
        universal_learner_eos = universal_learner_special_tokens.get("eos", "")
        self.universal_learner_format = universal_learner_bos + "{instruction}." + "\n{example}"  + universal_learner_eos
        self.lm_format = lm_special_tokens.get("bos", "") + "{text}" + lm_special_tokens.get("eos", "")
        self.mask_probability = mask_probability

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
        # Masking
        lm_input_ids = lm_encodings['input_ids']
        attention_mask = lm_encodings['attention_mask']
        labels = lm_input_ids.clone()
        padding_indices = attention_mask == 0
        labels[padding_indices] = self.label_pad_token_id
        if self.mask_probability > 0.0:
            masked_indices_for_attention = torch.rand(lm_input_ids.shape) < self.mask_probability
            attention_mask[masked_indices_for_attention] = 0
            masked_indices_for_input = torch.rand(lm_input_ids.shape) < self.mask_probability
            # ignore padding indices
            masked_indices_for_input[padding_indices] = False
            lm_input_ids[masked_indices_for_input] = self.lm_tokenizer.mask_token_id

        return {
            'universal_learner_input_ids': universal_learner_encodings['input_ids'], # (b, l)
            'universal_learner_attention_mask': universal_learner_encodings['attention_mask'], # (b, l)
            'lm_input_ids': lm_input_ids, # (b, l)
            'lm_attention_mask': attention_mask, # (b, l)
            'lm_labels': labels, # (b, l)
        }
    

