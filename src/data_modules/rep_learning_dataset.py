import bisect
import random
from typing import Dict, Tuple
import einops
import torch
from torch.utils.data.dataset import Dataset, ConcatDataset
from transformers import PreTrainedTokenizer, BatchEncoding
import datasets

from src.data_modules.constants import DATA


class RepLearningDataset(Dataset):
    def __init__(
            self,
            data_name: str,
            number_training_samples: int=1_000_000,
            neg_per_sample: int=1,
            pos_per_sample: int=1,
            seed: int=777,
            ):
        super().__init__()
        data_path = DATA[data_name]['data_path']
        self.data_name = data_name
        self.instruction = DATA[data_name]['instruction']
        self.enable_cross_batch_negative_sampling = DATA[data_name].get('enable_cross_batch_negative_sampling', True)
        self.number_training_samples = number_training_samples
        self.neg_per_sample = neg_per_sample
        self.pos_per_sample = pos_per_sample
        self.seed = seed
        try:
            data = datasets.load_dataset(data_name, split='train')
        except:
            data = datasets.load_dataset('json', data_files=data_path, split='train')
        if len(data) > number_training_samples:
            data = data.train_test_split(train_size=number_training_samples, seed=seed, shuffle=True)['train']
        self.data = data

        self.rng = random.Random(self.seed)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        pos = self.rng.sample(example['positive'], min(len(example['positive']), self.pos_per_sample))
        neg = self.rng.sample(example['negative'], min(len(example['negative']), self.neg_per_sample))
        assert len(pos) > 0, "At least one positive example per sample. Please check the data {}".format(self.data_name)
        assert len(neg) > 0, "At least one negative example per sample. Please check the data {}".format(self.data_name)
        return {
            'query_label': idx,
            'query': example['query'], # str
            'positive': pos, # list of str
            'negative': neg, # list of str
            'instruction': self.instruction,
            'enable_cross_batch_negative_sampling': self.enable_cross_batch_negative_sampling,
        }
    

class ConcatRepLearningDataset(ConcatDataset):
    """
    An extension of ConcatDataset that guarantees that each example has a unique query_label.
    """
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError(
                    "absolute value of index should not exceed dataset length"
                )
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        example =  self.datasets[dataset_idx][sample_idx]

        # Update the query_label to be unique across all datasets
        example['query_label'] = idx
        return example


class RepLearningCollator:
    def __init__(
            self,
            tokenizer: PreTrainedTokenizer,
            special_tokens: Dict[str, str],
            max_seq_length: int=512,
            label_pad_token_id: int=-100,
            ):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.label_pad_token_id = label_pad_token_id

        bos = special_tokens.get("bos", "")
        user_bos = special_tokens.get("user_bos", "")
        eos = special_tokens.get("eos", "")
        eot = special_tokens.get("eot", "")
        self.query_prompt = bos + user_bos + "{instruction}." 
        self.query_format = bos + user_bos + "{instruction}." + "\n{example}" + eot + eos
        self.candidate_prompt = bos + user_bos + "{instruction}. Candidate:" + "\n"
        self.candidate_format = bos + user_bos + "{instruction}. Candidate:" + "\n" + "{example}" + eot + eos
        

    def tokenize_example(
            self,
            example: str,
            is_query: bool,
            instruction: str="",
            ) -> BatchEncoding:
        if len(example) == 0:
            print('example:', example)
        if is_query:
            prompt = self.query_prompt.format(instruction=instruction)
            example = self.query_format.format(instruction=instruction, example=example)
        else:
            prompt = self.candidate_prompt.format(instruction=instruction)
            example = self.candidate_format.format(instruction=instruction, example=example)
        model_inputs = self.tokenizer(
            example,
            padding="longest",
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors=None,
            add_special_tokens=False, # already added in the format
        )
        # find the prompt length
        prompt_length = len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])
        try:
            assert len(model_inputs['input_ids']) > prompt_length, f"Input length is less than prompt length: {len(model_inputs['input_ids'])} <= {prompt_length}."
            model_inputs['prompt_length'] = prompt_length
        except:
            print('model_inputs:', model_inputs)
            print('example:', example)
            print('prompt:', prompt)

        return model_inputs
    
    def __call__(self, batch):
        batch_size = len(batch)
        
        query_labels = [example['query_label'] for example in batch]
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        # if all the examples in the batch have enable_cross_batch_negative_sampling==True, then the batch has enable_cross_batch_negative_sampling==True
        enable_cross_batch_negative_sampling = all([example['enable_cross_batch_negative_sampling'] for example in batch])

        min_pos_per_sample = min([len(example['positive']) for example in batch])
        min_neg_per_sample = min([len(example['negative']) for example in batch])
        breakpoint()
        assert min_pos_per_sample > 0, "At least one positive example per sample"
        assert min_neg_per_sample > 0, "At least one negative example per sample"
        batch_query = []
        batch_pos = []
        batch_neg = []
        for i, example in enumerate(batch):
            q = example['query']
            pos = example['positive']
            neg = example['negative']
            instruction = example['instruction']
    
            batch_query.append(self.tokenize_example(example=q, is_query=True, instruction=instruction))

            if len(pos) > min_pos_per_sample:
                pos = random.sample(pos, min_pos_per_sample) 
                pos = [(instruction, p) for p in pos]
            else:
                pos = [(instruction, p) for p in pos]
            for p in pos:
                assert len(p) == 2 and isinstance(p, tuple), f"Positive example must be a tuple of length 2. Got {p}"
                assert isinstance(p[1], str), f"Positive example must be a tuple of length 2. Got {p}"
                breakpoint()
                try:
                    batch_pos.append(self.tokenize_example(example=p[1], is_query=False, instruction=p[0]))
                except Exception as e:
                    print('Error:', e)
                    print('p:', p)

            if len(neg) > min_neg_per_sample:
                neg = random.sample(neg, min_neg_per_sample)
                neg = [(instruction, n) for n in neg]
            else:
                neg = [(instruction, n) for n in neg]
            for n in neg:
                assert len(n) == 2 and isinstance(n, tuple), f"Negative example must be a tuple of length 2. Got {n}"
                assert isinstance(n[1], str), f"Negative example must be a tuple of length 2. Got {n}"
                breakpoint()
                try:
                    batch_neg.append(self.tokenize_example(example=n[1], is_query=False, instruction=n[0]))
                except Exception as e:
                    print('Error:', e)
                    print('n:', n)

        len_q = len(batch_query)
        len_p = len(batch_pos)
        len_n = len(batch_neg)
        batch = batch_query + batch_pos + batch_neg
        batch = self.tokenizer.pad(batch, return_tensors='pt', pad_to_multiple_of=8) 

        query_input_ids = batch['input_ids'][:len_q]
        query_attention_mask = batch['attention_mask'][:len_q]
        query_prompt_length = batch['prompt_length'][:len_q]

        pos_input_ids = batch['input_ids'][len_q:len_q+len_p]
        pos_input_ids = einops.rearrange(pos_input_ids, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        pos_attention_mask = batch['attention_mask'][len_q:len_q+len_p]
        pos_attention_mask = einops.rearrange(pos_attention_mask, '(b n) l -> b n l', b=batch_size, n=min_pos_per_sample)
        pos_prompt_length = batch['prompt_length'][len_q:len_q+len_p]
        pos_prompt_length = einops.rearrange(pos_prompt_length, '(b n) -> b n', b=batch_size, n=min_pos_per_sample)

        neg_input_ids = batch['input_ids'][len_q+len_p:]
        neg_input_ids = einops.rearrange(neg_input_ids, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
        neg_attention_mask = batch['attention_mask'][len_q+len_p:]
        neg_attention_mask = einops.rearrange(neg_attention_mask, '(b n) l -> b n l', b=batch_size, n=min_neg_per_sample)
        neg_prompt_length = batch['prompt_length'][len_q+len_p:]
        neg_prompt_length = einops.rearrange(neg_prompt_length, '(b n) -> b n', b=batch_size, n=min_neg_per_sample)

        return {
            'enable_cross_batch_negative_sampling': enable_cross_batch_negative_sampling,
            'query_labels': query_labels, # (batch_size,)
            'query_input_ids': query_input_ids,
            'query_attention_mask': query_attention_mask,
            'query_prompt_length': query_prompt_length,
            'pos_input_ids': pos_input_ids,
            'pos_attention_mask': pos_attention_mask,
            'pos_prompt_length': pos_prompt_length,
            'neg_input_ids': neg_input_ids,
            'neg_attention_mask': neg_attention_mask,
            'neg_prompt_length': neg_prompt_length,
        }

            
