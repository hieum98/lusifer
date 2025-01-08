import os
import datasets

from lusifer.preprocessing.tool import load_nnlb_datasets, mine_hard_negatives


SELECTED_PAIRS = {
    'en-ru': 'eng_Latn-rus_Cyrl', 'en-zh': 'eng_Latn-zho_Hans', 
    'en-fr': 'eng_Latn-fra_Latn', 'en-es': 'eng_Latn-spa_Latn', 
    'en-it': 'eng_Latn-ita_Latn', 'en-nl': 'eng_Latn-nld_Latn', 
    'en-vi': 'eng_Latn-vie_Latn', 'en-id': 'eng_Latn-ind_Latn', 
    'en-hu': 'eng_Latn-hun_Latn', 'en-ro': 'eng_Latn-ron_Latn', 
    'en-sk': 'eng_Latn-slk_Latn', 'en-uk': 'eng_Latn-ukr_Cyrl', 
    'en-sr': 'eng_Latn-srp_Cyrl', 'en-hr': 'eng_Latn-hrv_Latn', 
    'en-hi': 'eng_Latn-hin_Deva', 'en-ta': 'eng_Latn-tam_Taml', 
    'en-ne': 'eng_Latn-npi_Deva', 'en-ml': 'eng_Latn-mal_Mlym', 
    'en-mr': 'eng_Latn-mar_Deva', 'ru-zh': 'rus_Cyrl-zho_Hans', 
    'ru-vi': 'rus_Cyrl-vie_Latn', 'ru-sk': 'rus_Cyrl-slk_Latn', 
    'ru-uk': 'rus_Cyrl-ukr_Cyrl', 'ru-sr': 'rus_Cyrl-srp_Cyrl', 
    'ru-ta': 'rus_Cyrl-tam_Taml', 'vi-zh': 'vie_Latn-zho_Hans'
    }

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/hieum/uonlp/lusifer/data/crosslingual')
    parser.add_argument('--corpus_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-large')

    args = parser.parse_args()
    data_dir = args.data_dir
    corpus_name = args.corpus_name
    model_name = args.model_name
    print(f'Processing {corpus_name}...')
    output_path = os.path.join(data_dir, f'{corpus_name}.jsonl')
    if os.path.exists(output_path):
        print(f'{corpus_name} has been processed!')
    else:
        data = load_nnlb_datasets(lang_pair=SELECTED_PAIRS[corpus_name])
        data.to_json(output_path, lines=True, orient='records')

    if os.path.exists(output_path.replace('.jsonl', '_hard_negatives.jsonl')):
        print(f'{corpus_name} hard negatives has been mined!')
    else:
        print(f'Mining hard negatives for {corpus_name}...')
        mine_hard_negatives(
                data_path=output_path, 
                model_name=model_name
            )

