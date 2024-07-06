import os
from itertools import combinations
import random
from typing import List
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import datasets

from src.preprocessing.minhash import dedup

def get_max_num_worker_suggest():
    # Get maximum number of processes in slurm task
    max_num_worker_suggest = 1
    try:
        max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
    num_proc = max_num_worker_suggest
    return num_proc


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def get_embeddings(model, tokenizer, texts, batchsize=16, instruction=''):
    texts = [f"{instruction} {text}" for text in texts]
    all_embeddings = []
    for i in range(0, len(texts), batchsize):
        batch = texts[i:i + batchsize]
        model_inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        device = next(model.parameters()).device
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        outputs = model(**model_inputs)
        embeddings = average_pool(outputs.last_hidden_state, model_inputs['attention_mask']) # (batch_size, hidden_size)
        embeddings = embeddings.cpu().numpy().tolist() # (batch_size, hidden_size)
        all_embeddings.extend(embeddings)
    return all_embeddings


def mine_hard_negatives(data_path, model_name='intfloat/e5-large'):
    num_proc = get_max_num_worker_suggest()
    def find_hard_negatives(example, corpus, model, tokenizer):
        query = example['query']
        positive_txt = example['positive']
        emb = get_embeddings(model=model, tokenizer=tokenizer, instruction='query:', texts=[query])
        emb = np.array(emb)
        
        negative = []
        scores, samples = corpus.get_nearest_examples('emb', emb, k=128)
        results = pd.DataFrame.from_dict(samples)
        results['score'] = scores
        results = results.sort_values(by='score', ascending=False)
        # Remove top-30 highest scores rows of the results to avoid false positives
        results = results.iloc[30:]
        for _, row in results.iterrows():
            negative.append(row["text"].strip())
        return {'query': query, 'positive': positive_txt, 'negative': negative}

    dataset = datasets.load_dataset('json', data_files=data_path, split='train')
    corpus = dataset.remove_columns(['query'])
    # Flatten the corpus 
    def flatten(batch):
        text = []
        for txt in batch['positive']:
            for x in txt:
                text.append(x)
        return {'text': text}
    corpus = corpus.map(lambda batch: flatten(batch), batched=True, remove_columns=corpus.column_names, num_proc=num_proc)
    corpus = dedup(
        column='text',
        data_path=None,
        num_proc=num_proc,
        ds=corpus,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.eval()
    model = model.to(0)
    
    corpus = corpus.map(
        lambda x: {'emb': get_embeddings(model=model, tokenizer=tokenizer, instruction='passage:', texts=x['text'])},
        batched=True,
        batch_size=32
    )
    corpus.add_faiss_index(column='emb')

    dataset = dataset.map(lambda x: find_hard_negatives(x, corpus, model, tokenizer))
    remove_columns = [n for n in dataset.column_names if n not in ['query', 'positive', 'negative']]
    dataset = dataset.remove_columns(remove_columns)

    dataset.to_json(data_path.replace('.jsonl', '_hard_negatives.jsonl'), lines=True, orient='records')


def mine_hard_negatives_for_clustering(data_path, model_name='intfloat/e5-large'):
    num_proc = get_max_num_worker_suggest()
    def find_hard_negatives(example, corpus, model, tokenizer):
        query = example['query']
        positive_txt = example['positive']
        pos_clusters = [str(x) for x in example['cluster']]
        emb = get_embeddings(model=model, tokenizer=tokenizer, instruction='query:', texts=[query])
        emb = np.array(emb)
        
        negative = []
        scores, samples = corpus.get_nearest_examples('emb', emb, k=2048)
        results = pd.DataFrame.from_dict(samples)
        results['score'] = scores
        results['cluster'] = results['cluster'].astype(str)
        # filter out positive samples
        results = results[~results['cluster'].isin(pos_clusters)]
        results = results.sort_values(by='score', ascending=False)
        # Remove top-30 highest scores rows of the results to avoid false positives
        results = results.iloc[20:140]
        for _, row in results.iterrows():
            negative.append(row["text"].strip())
        return {'query': query, 'positive': positive_txt, 'negative': negative}

    dataset = datasets.load_dataset('json', data_files=data_path, split='train')
    def flatten(batch):
        text = []
        cluster = []
        for q, txt, clus in zip(batch['query'], batch['positive'], batch['cluster']):
            text.append(q)
            cluster.append(clus)
            for x in txt:
                text.append(x)
                cluster.append(clus)
        return {'text': text, 'cluster': cluster}
    corpus = dataset.map(lambda x: flatten(x), remove_columns=dataset.column_names, num_proc=num_proc, batched=True)
    corpus = dedup(
        column='text',
        data_path=None,
        num_proc=num_proc,
        ds=corpus,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.eval()
    model = model.to(0)

    corpus = corpus.map(
        lambda x: {'emb': get_embeddings(model=model, tokenizer=tokenizer, instruction='passage:', texts=x['text'])},
        batched=True,
        batch_size=32
    )
    corpus.add_faiss_index(column='emb')

    dataset = dataset.map(lambda x: find_hard_negatives(x, corpus, model, tokenizer))
    remove_columns = [n for n in dataset.column_names if n not in ['query', 'positive', 'negative']]
    dataset = dataset.remove_columns(remove_columns)

    dataset.to_json(data_path.replace('.jsonl', '_hard_negatives.jsonl'), lines=True, orient='records')


def load_beir_dataset(dataset_path: str, **kwargs):
    # Get maximum number of processes in slurm task
    max_num_worker_suggest = 1
    try:
        max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
    num_proc = max_num_worker_suggest
    try:
        corpus = datasets.load_dataset('json', data_files=os.path.join(dataset_path, 'corpus.jsonl'), split='train')
    except:
        corpus = pd.read_json(os.path.join(dataset_path, 'corpus.jsonl'), lines=True)
        # remove all columns except 'text' and '_id'
        corpus = corpus[['text', '_id']]
        corpus = datasets.Dataset.from_pandas(corpus)
    try:
        corpus = dedup(
            column='text',
            data_path=None,
            num_proc=num_proc,
            ds=corpus,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
        # save corpus to jsonl file
        corpus.to_json(os.path.join(dataset_path, 'corpus_dedup.jsonl'), lines=True, orient='records')
    except:
        print("Deduplication failed for data in ", dataset_path)
        pass
    corpus = corpus.to_pandas()
    corpus['_id'] = corpus['_id'].astype(str)
    try:
        queries = datasets.load_dataset('json', data_files=os.path.join(dataset_path, 'queries.jsonl'), split='train')
    except:
        queries = pd.read_json(os.path.join(dataset_path, 'queries.jsonl'), lines=True)
        # remove all columns except 'text' and '_id'
        queries = queries[['text', '_id']]
        queries = datasets.Dataset.from_pandas(queries)
    try:
        queries = dedup(
            column='text',
            data_path=None,
            num_proc=num_proc,
            ds=queries,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in ", dataset_path)
        pass
    # get all .tsv files in the directory
    qrels_files = [f for f in os.listdir(os.path.join(dataset_path, 'qrels')) if f.endswith('.tsv')]
    # load qrels from all .tsv files in the directory 
    qrels = []
    for f in qrels_files:
        df = pd.read_csv(os.path.join(dataset_path, 'qrels', f), sep='\t', header=0)
        df = df[df['score'] > 0].drop(columns=['score'])
        qrels.append(df)
    # concatenate all qrels
    qrels = pd.concat(qrels)
    # group by query-id
    qrels = qrels.groupby('query-id')['corpus-id'].apply(list).reset_index()
    # convert query-id to string
    qrels['query-id'] = qrels['query-id'].astype(str)

    def process_example(example, qrels, corpus):
        q_id = example['_id']
        try:
            pos_ids = qrels[qrels['query-id'] == str(q_id)]['corpus-id'].values[0]
        except:
            pos_ids = []
        pos_ids = set(pos_ids)
        pos = []
        for id in pos_ids:
            try:
                x = corpus[corpus['_id'] == str(id)].iloc[0].to_dict()
            except:
                continue
            pos.append(f'{x["title"]}\n{x["text"]}')
        return {
            'query_id': q_id,
            'query': example['text'],
            'positive_ids': pos_ids,
            'positive': pos,
        }
    
    queries = queries.map(lambda x: process_example(x, qrels, corpus), num_proc=num_proc, remove_columns=queries.column_names)
    # filter out examples with no positive
    queries = queries.filter(lambda x: len(x['positive']) > 0)
    return queries


def load_pubmedqa_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        query = example['question']
        context = ' '.join(example['context']['contexts'])
        answer = example.get('final_decision', '') + '. ' if 'final_decision' in example else ''
        answer = answer + example['long_answer']
        return {
            'query': query,
            'positive': [context, answer],
            'full_answer': answer + ' ' + context,
        }
    subsets = ['pqa_artificial', 'pqa_labeled', 'pqa_unlabeled']
    all_data = []
    for subset in subsets:
        dataset = datasets.load_dataset('qiaojin/PubMedQA', subset, split='train')
        dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
        all_data.append(dataset)
    data = datasets.concatenate_datasets(all_data)
    try:
        data = dedup(
            column='full_answer',
            data_path=None,
            num_proc=num_proc,
            ds=data,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in PubMedQA")
        pass
    data = data.remove_columns(['full_answer'])
    return data


def load_pqa_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        return {
            'query': example['set'][0],
            'positive': [example['set'][1]],
            'full_positive': example['set'][1],
        }
    
    dataset = datasets.load_dataset('embedding-data/PAQ_pairs', split='train')
    dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='full_positive',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in PQA")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_snli_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        return {
            'query': example['premise'],
            'positive': [example['hypothesis']],
            'full_positive': example['hypothesis'],
        }
    
    dataset = datasets.load_dataset('stanfordnlp/snli')
    dataset = datasets.concatenate_datasets(dataset.values())
    # only keep the label entailment
    dataset = dataset.filter(lambda x: x['label'] == 0, num_proc=num_proc)
    dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='full_positive',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except: 
        print("Deduplication failed for data in SNLI")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_squad_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        return {
            'query': example['question'],
            'positive': [example['context']],
            'full_positive': example['context'],
        }
    
    dataset = datasets.load_dataset('rajpurkar/squad')
    dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in SQUAD")
        pass
    dataset = dataset.remove_columns(['full_positive'])

    return dataset


def load_sts_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        return {
            'query': example['sentence1'],
            'positive': [example['sentence2']],
            'full_positive': example['sentence2'],
        }
    
    dataset = []
    dataset.extend(datasets.load_dataset('mteb/sts12-sts').values())
    dataset.extend(datasets.load_dataset('mteb/sts22-crosslingual-sts', 'en').values())
    dataset.extend(datasets.load_dataset('mteb/stsbenchmark-sts').values())
    new_dataset = []
    for ds in dataset:
        # Filter out examples with with score < 3
        ds = ds.filter(lambda x: x['score'] >= 3, num_proc=num_proc)
        # remove all columns except 'sentence1', 'sentence2'
        remove_columns = [col for col in ds.column_names if col not in ['sentence1', 'sentence2']]
        ds = ds.remove_columns(remove_columns)
        new_dataset.append(ds)
    dataset = datasets.concatenate_datasets(new_dataset)
    dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
        dataset = dedup(
            column='full_positive',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in STS")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_amazon_counterfactual_dataset(language: str = 'en', **kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        all_labels = ['not-counterfactual', 'counterfactual']
        negative = [x for x in all_labels if x != example['label_text']]
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': negative,
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/amazon_counterfactual', language)
    dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in Amazon Counterfactual")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_amazon_review_dataset(language: str = 'en', **kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example):
        label_id_to_text = {
            0: 'Poor',
            1: 'Fair',
            2: 'Good',
            3: 'Very good',
            4: 'Excellent',
        }
        label = label_id_to_text[example['label']]
        negative = set(label_id_to_text.values()) - set([label])
        return {
            'query': example['text'],
            'positive': [label],
            'negative': list(negative),
            'full_positive': label,
        }
    
    dataset = datasets.load_dataset('mteb/amazon_reviews_multi', language)
    dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset.map(lambda x: process_example(x), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in Amazon Review")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_banking77_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example, all_labels: List[str]):
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': list(all_labels - set([example['label_text']])),
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/banking77')
    dataset = datasets.concatenate_datasets(dataset.values())
    all_labels = set(dataset['label_text'])
    dataset = dataset.map(lambda x: process_example(x, all_labels=all_labels), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in Banking77")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_emotion_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example, all_labels: List[str]):
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': list(all_labels - set([example['label_text']])),
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/emotion')
    dataset = datasets.concatenate_datasets(dataset.values())
    all_labels = set(dataset['label_text'])
    dataset = dataset.map(lambda x: process_example(x, all_labels=all_labels), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in Emotion")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_imdb_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example, all_labels: List[str]):
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': list(all_labels - set([example['label_text']])),
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/imdb')
    dataset = datasets.concatenate_datasets(dataset.values())
    all_labels = set(dataset['label_text'])
    dataset = dataset.map(lambda x: process_example(x, all_labels=all_labels), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in IMDB")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_mtop_intent_dataset(language: str='en', **kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example, all_labels: List[str]):
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': list(all_labels - set([example['label_text']])),
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/mtop_intent', language)
    dataset = datasets.concatenate_datasets(dataset.values())
    all_labels = set(dataset['label_text'])
    dataset = dataset.map(lambda x: process_example(x, all_labels=all_labels), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in MTop Intent")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_toxic_conversations_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example, all_labels: List[str]):
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': list(all_labels - set([example['label_text']])),
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/toxic_conversations_50k')
    dataset = datasets.concatenate_datasets(dataset.values())
    all_labels = set(dataset['label_text'])
    dataset = dataset.map(lambda x: process_example(x, all_labels=all_labels), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in Toxic Conversations")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_tweet_sentiments_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_example(example, all_labels: List[str]):
        return {
            'query': example['text'],
            'positive': [example['label_text']],
            'negative': list(all_labels - set([example['label_text']])),
            'full_positive': example['label_text'],
        }
    
    dataset = datasets.load_dataset('mteb/tweet_sentiment_extraction')
    dataset = datasets.concatenate_datasets(dataset.values())
    all_labels = set(dataset['label_text'])
    dataset = dataset.map(lambda x: process_example(x, all_labels=all_labels), num_proc=num_proc, remove_columns=dataset.column_names)
    try:
        dataset = dedup(
            column='query',
            data_path=None,
            num_proc=num_proc,
            ds=dataset,
            batch_size=100_000,
            idx_column=None,
            ngram=5,
            min_length=5,
            num_perm=1000,
            threshold=0.8,
        )
    except:
        print("Deduplication failed for data in Tweet Sentiments")
        pass
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


def load_arxiv_s2s_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        titles = batch['title']
        cluster = batch['categories']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_title, clus_name in zip(titles, cluster):
            all_collections = [cluster_title[i:i+32] for i in range(0, len(cluster_title), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                clus = clus_name.split(' ')
                clus = [x.strip().split('.')[0].strip() for x in clus]
                clus = '; '.join(list(set(clus)))
                examples['cluster'].append(clus)
        return examples

    dataset = datasets.load_dataset('mteb/raw_arxiv', split='train')
    remove_columns = [col for col in dataset.column_names if col not in ['title', 'categories']]
    dataset = dataset.remove_columns(remove_columns)
    dataset = dataset.filter(lambda x: len(x['title']) > 20, num_proc=num_proc)
    dataset = dedup(
        column='title',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    df = dataset.to_pandas()
    # group by categories
    df = df.groupby('categories')['title'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of titles < 4
    dataset = dataset.filter(lambda x: len(x['title']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'title': x['title'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, remove_columns=dataset.column_names, num_proc=num_proc)
    return dataset


def load_arxiv_p2p_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        abstracts = batch['abstract']
        cluster = batch['categories']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_abstract, clus_name in zip(abstracts, cluster):
            all_collections = [cluster_abstract[i:i+32] for i in range(0, len(cluster_abstract), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                clus = clus_name.split(' ')
                clus = [x.strip().split('.')[0].strip() for x in clus]
                clus = '; '.join(list(set(clus)))
                examples['cluster'].append(clus)
        return examples
    
    dataset = datasets.load_dataset('mteb/raw_arxiv', split='train')
    remove_columns = [col for col in dataset.column_names if col not in ['title', 'abstract', 'categories']]
    dataset = dataset.remove_columns(remove_columns)
    dataset = dataset.map(lambda x: {'abstract': x['title'] + '\n' + x['abstract'], 'categories': x['categories']}, num_proc=num_proc)
    # filter out examples with length < 5 and length > 2000
    dataset = dataset.filter(lambda x: len(x['abstract']) > 100 and len(x['abstract']) < 2000, num_proc=num_proc)
    dataset = dedup(
        column='abstract',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    df = dataset.to_pandas()
    # group by categories
    df = df.groupby('categories')['abstract'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of titles < 4
    dataset = dataset.filter(lambda x: len(x['abstract']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'abstract': x['abstract'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, batch_size=100, num_proc=num_proc, remove_columns=dataset.column_names)
    return dataset


def load_bio_s2s_dataset(data_name: str, **kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        titles = batch['title']
        cluster = batch['category']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_title, clus in zip(titles, cluster):
            all_collections = [cluster_title[i:i+32] for i in range(0, len(cluster_title), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                examples['cluster'].append(clus)
        return examples

    dataset = datasets.load_dataset(data_name, split='train')
    remove_columns = [col for col in dataset.column_names if col not in ['title', 'category']]
    dataset = dataset.remove_columns(remove_columns)
    dataset = dataset.filter(lambda x: len(x['title']) > 20, num_proc=num_proc)
    dataset = dedup(
        column='title',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    df = dataset.to_pandas()
    # group by category
    df = df.groupby('category')['title'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of titles < 4
    dataset = dataset.filter(lambda x: len(x['title']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'title': x['title'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)
    return dataset


def load_bio_p2p_dataset(data_name: str, **kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        abstracts = batch['abstract']
        cluster = batch['category']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_abstract, clus in zip(abstracts, cluster):
            all_collections = [cluster_abstract[i:i+32] for i in range(0, len(cluster_abstract), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                examples['cluster'].append(clus)
        return examples
    
    dataset = datasets.load_dataset(data_name, split='train')
    remove_columns = [col for col in dataset.column_names if col not in ['title', 'abstract', 'category']]
    dataset = dataset.remove_columns(remove_columns)
    dataset = dataset.map(lambda x: {'abstract': x['title'] + '\n' + x['abstract'], 'category': x['category']}, num_proc=num_proc)
    # filter out examples with length < 5 and length > 2000
    dataset = dataset.filter(lambda x: len(x['abstract']) > 100 and len(x['abstract']) < 2000, num_proc=num_proc)
    dataset = dedup(
        column='abstract',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    df = dataset.to_pandas()
    # group by category
    df = df.groupby('category')['abstract'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of abstracts < 4
    dataset = dataset.filter(lambda x: len(x['abstract']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'abstract': x['abstract'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)
    return dataset


def load_20newsgroups_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        txts = batch['text']
        cluster = batch['label']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_txt, clus in zip(txts, cluster):
            all_collections = [cluster_txt[i:i+32] for i in range(0, len(cluster_txt), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                examples['cluster'].append(clus)
        return examples
    
    dataset = datasets.load_dataset('SetFit/20_newsgroups')
    dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset.remove_columns(['label_text'])
    # filter out examples with length < 5 and length > 2000
    dataset = dataset.filter(lambda x: len(x['text']) > 100 and len(x['text']) < 2000, num_proc=num_proc)
    dataset = dedup(
        column='text',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    # group by label
    df = dataset.to_pandas()
    df = df.groupby('label')['text'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of texts < 4
    dataset = dataset.filter(lambda x: len(x['text']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'text': x['text'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)
    return dataset


def flatten_dataset(batch):
    text = []
    labels = []
    for txt, label in zip(batch['sentences'], batch['labels']):
        for x, l in zip(txt, label):
            text.append(x)
            labels.append(l)
    return {'text': text, 'label': labels}


def load_reddit_clustering_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        txts = batch['text']
        cluster = batch['label']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_txt, clus in zip(txts, cluster):
            all_collections = [cluster_txt[i:i+32] for i in range(0, len(cluster_txt), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                examples['cluster'].append(clus)
        return examples
    
    dataset = datasets.load_dataset('mteb/reddit-clustering-p2p')
    dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset.map(flatten_dataset, batch_size=1,
                          batched=True, num_proc=num_proc, remove_columns=dataset.column_names)

    dataset = dataset.filter(lambda x: len(x['text']) > 50 and len(x['text']) < 2000, num_proc=num_proc)
    dataset = dedup(
        column='text',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    df = dataset.to_pandas()
    # group by label
    df = df.groupby('label')['text'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of texts < 4
    dataset = dataset.filter(lambda x: len(x['text']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'text': x['text'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)
    return dataset


def load_stackexchange_clustering_dataset(**kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        txts = batch['text']
        cluster = batch['label']
        examples = {
            'query': [],
            'positive': [],
            'cluster': []
        }
        for cluster_txt, clus in zip(txts, cluster):
            all_collections = [cluster_txt[i:i+32] for i in range(0, len(cluster_txt), 32)]
            collections = [all_collections[i] for i in random.sample(range(len(all_collections)), min(16, len(all_collections)))]
            for collection in collections:
                examples['query'].append(collection[0])
                positive = collection[1:]
                examples['positive'].append(positive)
                examples['cluster'].append(clus)
        return examples
    
    dataset = datasets.load_dataset('mteb/stackexchange-clustering-p2p')
    dataset = datasets.concatenate_datasets(dataset.values())
    dataset = dataset.map(flatten_dataset, batched=True, batch_size=1, num_proc=num_proc, remove_columns=dataset.column_names)

    dataset = dataset.filter(lambda x: len(x['text']) > 50 and len(x['text']) < 2000, num_proc=num_proc)
    dataset = dedup(
        column='text',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    df = dataset.to_pandas()
    # group by label
    df = df.groupby('label')['text'].apply(list).reset_index()
    dataset = datasets.Dataset.from_pandas(df)
    # filter out examples with number of texts < 4
    dataset = dataset.filter(lambda x: len(x['text']) > 4, num_proc=num_proc)
    dataset = dataset.map(lambda x: {'text': x['text'][:3000]}, num_proc=num_proc)
    dataset = dataset.map(process_batch, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)
    return dataset


def load_nnlb_datasets(lang_pair: str, **kwargs):
    num_proc = get_max_num_worker_suggest()
    def process_batch(batch):
        pairs = batch['translation']
        query = []
        positive = []
        full_positive = []
        for p in pairs:
            p = p.values()
            if random.random() > 0.5:
                query.append(p[0])
                positive.append([p[1]])
                full_positive.append(p[1])
            else:
                query.append(p[1])
                positive.append([p[0]])
                full_positive.append(p[0])

        return {'query': query, 'positive': positive, 'full_positive': full_positive}

    dataset = datasets.load_dataset('nnlb', lang_pair, split='train')
    dataset = dataset.map(process_batch, batched=True, num_proc=num_proc, remove_columns=dataset.column_names)

    dataset = dedup(
        column='query',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    dataset = dedup(
        column='full_positive',
        data_path=None,
        num_proc=num_proc,
        ds=dataset,
        batch_size=100_000,
        idx_column=None,
        ngram=5,
        min_length=5,
        num_perm=1000,
        threshold=0.8,
    )
    dataset = dataset.remove_columns(['full_positive'])
    return dataset


