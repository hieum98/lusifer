import os
import pandas as pd
import numpy as np
import datasets
from transformers import AutoTokenizer, AutoModel

from src.preprocessing.tool import load_beir_dataset, get_embeddings


def parse_beir_format(data_dir, name, output_path):
    # get all subdirectories in the directory
    dataset_path = os.path.join(data_dir, name)
    subcorpus = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f != 'qrels']
    if len(subcorpus) != 0:
        corpus = [load_beir_dataset(os.path.join(dataset_path, f)) for f in subcorpus]
        # concatenate all subcorpus
        corpus = datasets.concatenate_datasets(corpus)
    else:
        corpus = load_beir_dataset(dataset_path)
    # save corpus to jsonl file
    corpus.to_json(output_path, lines=True, orient='records')


def mine_hard_negatives(data_dir, name, data_path, model_name='intfloat/e5-large'):
    def find_hard_negatives(example, corpus, model, tokenizer):
        query = example['query']
        positive_txt = example['positive']
        positive_ids = [str(x) for x in example['positive_ids']]
        emb = get_embeddings(model=model, tokenizer=tokenizer, instruction='query:', texts=[query])
        emb = np.array(emb)
        
        negative = []
        scores, samples = corpus.get_nearest_examples('emb', emb, k=128)
        results = pd.DataFrame.from_dict(samples)
        results['score'] = scores
        results['_id'] = results['_id'].astype(str)

        # filter out positive samples
        results = results[~results['_id'].isin(positive_ids)]
        results = results.sort_values(by='score', ascending=False)
        # Remove top-30 highest scores rows of the results to avoid false positives
        results = results.iloc[30:]
        for _, row in results.iterrows():
            negative.append(f'{row["title"]} {row["text"]}'.strip())
        return {'query': query, 'positive': positive_txt, 'negative': negative}
        
    dataset = datasets.load_dataset('json', data_files=data_path, split='train')
    # load all corpus_data
    # get all subdirectories in the directory
    dataset_path = os.path.join(data_dir, name)
    subcorpus = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f != 'qrels']
    if len(subcorpus) != 0:
        all_corpus = []
        for f in subcorpus:
            corpus = datasets.load_dataset('json', data_files=os.path.join(dataset_path, f, 'corpus_dedup.jsonl'), split='train')
            all_corpus.append(corpus)
            corpus = datasets.concatenate_datasets(all_corpus)
    else:
        corpus = datasets.load_dataset('json', data_files=os.path.join(dataset_path, 'corpus_dedup.jsonl'), split='train[:1000000]')

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.eval()
    model = model.to(0)
    
    corpus = corpus.map(
        lambda x: {'emb': get_embeddings(model=model, tokenizer=tokenizer, instruction='passage:', texts=[f"{t} {txt}".strip() for t, txt in zip(x['title'], x['text'])])},
        batched=True,
        batch_size=32
    )
    corpus.add_faiss_index(column='emb')

    dataset = dataset.map(lambda x: find_hard_negatives(x, corpus, model, tokenizer))
    remove_columns = [n for n in dataset.column_names if n not in ['query', 'positive', 'negative']]
    dataset = dataset.remove_columns(remove_columns)

    dataset.to_json(data_path.replace('.jsonl', '_hard_negatives.jsonl'), lines=True, orient='records')

if __name__ == '__main__':
    import argparse

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/hieum/uonlp/lusifer/data')
    parser.add_argument('--corpus_name', type=str, required=True)
    args = parser.parse_args()

    # corpus_name = ['trec-covid', 'nq', 'webis-touche2020', 'scidocs', 'dbpedia-entity', 
    #                'nfcorpus', 'arguana', 'fever', 'cqadupstack', 'hotpotqa', 'fiqa', 'msmarco', 'scifact', 'quora']
    data_dir = args.data_dir
    name = args.corpus_name
    
    print(f'Processing {name}...')
    output_path = os.path.join(data_dir, 'beir', f'{name}.jsonl')
    if os.path.exists(output_path):
        print(f'{name} has been processed!')
    else:
        print(f'Parsing {name}...')
        parse_beir_format(
            data_dir=data_dir,
            name=name,
            output_path=output_path
        )
    
    if os.path.exists(output_path.replace('.jsonl', '_hard_negatives.jsonl')):
        print(f'{name} hard negatives has been mined!')
    else:
        print(f'Mining hard negatives for {name}...')
        mine_hard_negatives(
            data_dir=data_dir,
            name=name,
            data_path=output_path
        )

    
        


