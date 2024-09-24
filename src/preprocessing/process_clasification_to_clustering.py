import os
import random
import datasets
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel

from src.preprocessing.tool import get_embeddings


DATA_NAMES = [
    'Hieuman/Amazon-Counterfactual', 'Hieuman/Amazon-Review', 'Hieuman/Banking77', 'Hieuman/Emotion',
    'Hieuman/imdb', 'Hieuman/MTOP-Intent', 'Hieuman/Toxic-conversation', 'Hieuman/Tweet-Sentiment'
]

max_num_worker_suggest = 1
try:
    max_num_worker_suggest = len(os.sched_getaffinity(0))
except Exception:
    pass
num_workers = min(1, max_num_worker_suggest)


def process_batch(batch):
    txts = batch['query']
    cluster = batch['label']
    examples = {
        'query': [],
        'positive': [],
        'cluster': []
    }
    for cluster_txt, clus in zip(txts, cluster):
        for i, txt in enumerate(cluster_txt):
            positive = cluster_txt[:i] + cluster_txt[i+1:]
            random.shuffle(positive)
            num_positive = min(16, len(positive))
            if len(positive) > num_positive:
                start = random.randint(0, len(positive)-num_positive)
                positive = positive[start:start+num_positive]
            examples['query'].append(txt)
            examples['positive'].append(positive)
            examples['cluster'].append(clus)
    return examples


def mine_hard_negatives_for_clustering(dataset, model_name='intfloat/e5-large'):
    num_proc = num_workers

    def find_hard_negatives(example, corpus, model, tokenizer):
        query = example['query']
        positive_txt = example['positive']
        pos_clusters = str(example['cluster'])
        emb = get_embeddings(model=model, tokenizer=tokenizer, instruction='query:', texts=[query])
        emb = np.array(emb)
        
        negative = []
        scores, samples = corpus.get_nearest_examples('emb', emb, k=2048)
        results = pd.DataFrame.from_dict(samples)
        results['score'] = scores
        results['cluster'] = results['cluster'].astype(str)
        # filter out positive sample
        results = results[results['cluster'] != pos_clusters]
        results = results.sort_values(by='score', ascending=False)
        # Remove top-30 highest scores rows of the results to avoid false positives
        results = results.iloc[20:140]
        for _, row in results.iterrows():
            negative.append(row["text"].strip())
        return {'query': query, 'positive': positive_txt, 'negative': negative}

    corpus = dataset.map(lambda x: {'text': x['query'], 'cluster': x['cluster']}, remove_columns=dataset.column_names, num_proc=num_proc, batched=True)
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

    return dataset


def process_clasification_to_clustering(data_name):
    data = datasets.load_dataset(data_name, split='train')
    # Check whether len of positive is 1 for all data, if not, raise error
    data = data.map(lambda x: {'len_positive': len(x['positive'])}, num_proc=num_workers)
    if len(data.unique('len_positive')) != 1 or data.unique('len_positive')[0] != 1:
        raise ValueError(f'len_positive is not 1 for all data in {data_name}')
    data = data.map(lambda x: {'label': x['positive'][0]}, num_proc=num_workers)
    collumns_to_remove = [col for col in data.column_names if col not in ['label', 'query']]
    data = data.remove_columns(collumns_to_remove)

    # Group by label
    data = data.to_pandas()
    data = data.groupby('label')['query'].apply(list).reset_index()
    data = datasets.Dataset.from_pandas(data)

    # Process data
    data = data.map(process_batch, num_proc=num_workers, batched=True, remove_columns=data.column_names)
    data = mine_hard_negatives_for_clustering(data, model_name='intfloat/e5-large')
    return data


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='Hieuman/Amazon-Counterfactual')
    parser.add_argument("--push_to_hub", action='store_true')
    args = parser.parse_args()

    data = process_clasification_to_clustering(args.data_name)
    if args.push_to_hub:
        data.push_to_hub(args.data_name+'_clustering')

