import os
from functools import partial
import datasets

from lusifer.preprocessing.tool import (
    load_20newsgroups_dataset,
    load_amazon_counterfactual_dataset,
    load_amazon_review_dataset,
    load_arxiv_p2p_dataset,
    load_arxiv_s2s_dataset,
    load_banking77_dataset,
    load_bio_p2p_dataset,
    load_bio_s2s_dataset,
    load_emotion_dataset,
    load_imdb_dataset,
    load_mtop_intent_dataset,
    load_pubmedqa_dataset,
    load_pqa_dataset,
    load_reddit_clustering_dataset,
    load_snli_dataset,
    load_squad_dataset,
    load_stackexchange_clustering_dataset,
    load_sts_dataset,
    load_toxic_conversations_dataset,
    load_tweet_sentiments_dataset,
    mine_hard_negatives,
    mine_hard_negatives_for_clustering
)   


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/hieum/uonlp/lusifer/data')
    parser.add_argument('--corpus_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='intfloat/e5-large')
    parser.add_argument('--language', type=str, default='en')
    args = parser.parse_args()
    data_dir = args.data_dir
    corpus_name = args.corpus_name
    model_name = args.model_name
    language = args.language
    print(f'Processing {corpus_name}...')

    if corpus_name == 'pubmedqa':
        output_path = os.path.join(data_dir, 'pubmedqa.jsonl')
        data_loader_fn = load_pubmedqa_dataset
    elif corpus_name == 'pqa':
        output_path = os.path.join(data_dir, 'pqa.jsonl')
        data_loader_fn = load_pqa_dataset
    elif corpus_name == 'snli':
        output_path = os.path.join(data_dir, 'snli.jsonl')
        data_loader_fn = load_snli_dataset
    elif corpus_name == 'squad':
        output_path = os.path.join(data_dir, 'squad.jsonl')
        data_loader_fn = load_squad_dataset
    elif corpus_name == 'sts':
        output_path = os.path.join(data_dir, 'sts.jsonl')
        data_loader_fn = load_sts_dataset
    elif corpus_name == 'amazon_counterfactual':
        output_path = os.path.join(data_dir, 'amazon_counterfactual.jsonl')
        data_loader_fn = partial(load_amazon_counterfactual_dataset, language=language)
    elif corpus_name == 'amazon_review':
        output_path = os.path.join(data_dir, 'amazon_review.jsonl')
        data_loader_fn = partial(load_amazon_review_dataset, language=language)
    elif corpus_name == 'banking77':
        output_path = os.path.join(data_dir, 'banking77.jsonl')
        data_loader_fn = load_banking77_dataset
    elif corpus_name == 'emotion':
        output_path = os.path.join(data_dir, 'emotion.jsonl')
        data_loader_fn = load_emotion_dataset
    elif corpus_name == 'imdb':
        output_path = os.path.join(data_dir, 'imdb.jsonl')
        data_loader_fn = load_imdb_dataset
    elif corpus_name == 'mtop_intent':
        output_path = os.path.join(data_dir, 'mtop_intent.jsonl')
        data_loader_fn = load_mtop_intent_dataset
    elif corpus_name == 'toxic_conversations':
        output_path = os.path.join(data_dir, 'toxic_conversations.jsonl')
        data_loader_fn = load_toxic_conversations_dataset
    elif corpus_name == 'tweet_sentiments':
        output_path = os.path.join(data_dir, 'tweet_sentiments.jsonl')
        data_loader_fn = load_tweet_sentiments_dataset
    elif corpus_name == 'arxivS2S':
        output_path = os.path.join(data_dir, 'arxivS2S.jsonl')
        data_loader_fn = load_arxiv_s2s_dataset
    elif corpus_name == 'arxivP2P':
        output_path = os.path.join(data_dir, 'arxivP2P.jsonl')
        data_loader_fn = load_arxiv_p2p_dataset
    elif corpus_name == 'biorxivS2S':
        output_path = os.path.join(data_dir, 'biorxivS2S.jsonl')
        data_loader_fn = partial(load_bio_s2s_dataset, data_name='mteb/raw_biorxiv')
    elif corpus_name == 'biorxivP2P':
        output_path = os.path.join(data_dir, 'biorxivP2P.jsonl')
        data_loader_fn = partial(load_bio_p2p_dataset, data_name='mteb/raw_biorxiv')
    elif corpus_name == 'medrxivS2S':
        output_path = os.path.join(data_dir, 'medrxivS2S.jsonl')
        data_loader_fn = partial(load_bio_s2s_dataset, data_name='mteb/raw_medrxiv')
    elif corpus_name == 'medrxivP2P':
        output_path = os.path.join(data_dir, 'medrxivP2P.jsonl')
        data_loader_fn = partial(load_bio_p2p_dataset, data_name='mteb/raw_medrxiv')
    elif corpus_name == '20newsgroups':
        output_path = os.path.join(data_dir, '20newsgroups.jsonl')
        data_loader_fn = load_20newsgroups_dataset
    elif corpus_name == 'reddit_clustering':
        output_path = os.path.join(data_dir, 'reddit_clustering.jsonl')
        data_loader_fn =  load_reddit_clustering_dataset
    elif corpus_name == 'stackexchange_clustering':
        output_path = os.path.join(data_dir, 'stackexchange_clustering.jsonl')
        data_loader_fn =  load_stackexchange_clustering_dataset

    if os.path.exists(output_path):
        print(f'{corpus_name} has been processed!')
    else:
        print(f'Loading {corpus_name} data...')
        data = data_loader_fn()
        data.to_json(output_path, lines=True, orient='records')

    if os.path.exists(output_path.replace('.jsonl', '_hard_negatives.jsonl')):
        print(f'{corpus_name} hard negatives has been mined!')
    else:
        print(f'Mining hard negatives for {corpus_name}...')
        if corpus_name in ['pubmedqa', 'pqa', 'snli', 'squad', 'sts']:
            mine_hard_negatives(
                data_path=output_path, 
                model_name=model_name
            )
        elif corpus_name in ['arxivS2S', 'arxivP2P', 'biorxivS2S', 'biorxivP2P', 'medrxivS2S', 'medrxivP2P', '20newsgroups', 'reddit_clustering', 'stackexchange_clustering']:
            mine_hard_negatives_for_clustering(
                data_path=output_path, 
                model_name=model_name
            )
        else:
            print(f'No hard negatives mining for {corpus_name}!')
            data = datasets.load_dataset('json', data_files=output_path, split='train')
            data.to_json(output_path.replace('.jsonl', '_hard_negatives.jsonl'), lines=True, orient='records')



