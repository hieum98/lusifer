import argparse
import gc
import math
import os
import random
import datasets
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import faiss


# Get maximum number of processes in slurm task
max_num_worker_suggest = 1
try:
    max_num_worker_suggest = len(os.sched_getaffinity(0))
except Exception:
    pass
num_proc = max_num_worker_suggest


def average_pool(last_hidden_states: torch.Tensor,
                 attention_mask: torch.Tensor) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def add_embedding(example, model, tokenizer, text_key='query', embed_key='query_embedding'):
    text_collection = example[text_key]
    if isinstance(text_collection, str):
        text_collection = [text_collection]
    embeddings = []
    for i in range(0, len(text_collection), 64):
        batch = text_collection[i:i+64]
        batch = [f'query: {text}' if not isinstance(text, list) else f'query: {text[0]}' for text in batch]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        inputs = {key: inputs[key].to(device) for key in inputs}
        with torch.no_grad():
            outputs = model(**inputs)
            emb = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            embeddings.append(emb) # (batch_size, hidden_size)
    embeddings = torch.cat(embeddings, dim=0) # (len(texts), hidden_size)
    # Normalize the embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    embeddings = embeddings.cpu().numpy().tolist()
    return {embed_key: embeddings}


def clustering(
        data: datasets.Dataset,
        ncentroids: int,
        niter: int=100,
        verbose: bool=False,
        embed_key: str='query_embedding'
        ):
    
    data_size = len(data)
    if data_size < ncentroids * 20:
        ncentroids = data_size // 20
    # get the embeddings matrix of the dataset
    embeddings = data[embed_key]
    embeddings = np.array(embeddings)
    d = embeddings.shape[1]
    # clustering
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, min_points_per_centroid=20)
    kmeans.train(embeddings)
    _, I = kmeans.index.search(embeddings, 1)
    # assign the cluster to each example
    data = data.map(lambda x, idx: {"cluster": int(I[idx])}, with_indices=True, num_proc=num_proc)
    return data


if __name__=='__main__':
    # add arguments
    parser = argparse.ArgumentParser(description='Clustering and retrieve hard examples')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    args = parser.parse_args()

    ncentroids = 1024  # number of clusters
    niter = 200        # number of iterations
    verbose = True     # verbose output
    number_data = 20_000  # number of data to reduce

    # Load the model
    encoder_model_name_or_path = 'intfloat/multilingual-e5-large'
    tokenizer = AutoTokenizer.from_pretrained(encoder_model_name_or_path)
    model = AutoModel.from_pretrained(encoder_model_name_or_path)
    # Check cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    # Load the dataset
    dataset = datasets.load_dataset(args.dataset, split='train')
    dataset = dataset.map(lambda x: add_embedding(x, model, tokenizer), batched=True)
    dataset = clustering(dataset, ncentroids, niter, verbose)
    # Remove the query_embedding key
    dataset = dataset.remove_columns('query_embedding')

    # Update the dataset to hub
    dataset.push_to_hub(args.dataset, private=False)

    # Reduce the data by using the cluster
    cluster = set(dataset['cluster'])
    if len(dataset) > number_data:
        example_per_cluster = math.ceil(number_data / len(cluster))
        cluster_with_id = dataset.map(lambda example, idx: {'id': idx, 'cluster': example['cluster']}, with_indices=True, num_proc=num_proc, remove_columns=dataset.column_names)
        cluster_with_id = cluster_with_id.to_pandas()
        # group by cluster
        cluster_with_id = cluster_with_id.groupby('cluster')['id'].apply(list).reset_index()
        cluster_with_id = cluster_with_id.to_dict(orient='records')

        # get the examples
        selected_index = []
        for clus in cluster_with_id:
            in_cluster_index = clus['id']
            in_cluster_index = random.sample(in_cluster_index, min(len(in_cluster_index), example_per_cluster))
            selected_index.extend(in_cluster_index)
        
        if len(selected_index) < number_data:
            all_data_index = list(range(len(dataset)))
            random.shuffle(all_data_index)
            for idx in all_data_index:
                if idx not in selected_index:
                    selected_index.append(idx)
                if len(selected_index) >= number_data:
                    break
        
        dataset = dataset.select(selected_index)
        # Update the dataset to hub
        name = f'{args.dataset}-reduced'
        dataset.push_to_hub(name, private=False)

    
