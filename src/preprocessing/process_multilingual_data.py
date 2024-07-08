import os

from src.preprocessing.process_beir import mine_hard_negatives, parse_beir_format
from src.preprocessing.tool import load_miracl_datasets, load_mr_tidy_datasets


ALL_CORPUS = {
    'mr-tidy': [
        {'arabic': 'ar'}, {'bengali': 'bn'}, {'english': 'en'}, {'indonesian': 'id'}, 
        {'finnish': 'fi'}, {'korean': 'ko'}, {'russian':'ru'}, {'swahili': 'sw'}, 
        {'telugu': 'te'}, {'thai': 'th'}, {'japanese': 'ja'},
        ],
    'miracl': ['ar', 'bn', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'zh', 'de', 'yo'],
    'vihealthqa': ['vi']
}


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/home/hieum/uonlp/lusifer/data')
    parser.add_argument('--corpus_name', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='intfloat/multilingual-e5-large')
    args = parser.parse_args()

    data_dir = args.data_dir
    corpus_name = args.corpus_name
    model_name = args.model_name

    languages = ALL_CORPUS[corpus_name]
    if corpus_name == 'mr-tidy':
        for lang in languages:
            lang_name = list(lang.keys())[0]
            lang = lang[lang_name]
            print(f'Processing {corpus_name} in {lang}...')
            output_path = os.path.join(data_dir, lang, f'{corpus_name}_hard_negatives.jsonl')
            if os.path.exists(output_path):
                print(f'{corpus_name} has been processed!')
            else:
                data = load_mr_tidy_datasets(lang=lang_name)
                data.to_json(output_path, lines=True, orient='records')
    elif corpus_name == 'miracl':
        for lang in languages:
            print(f'Processing {corpus_name} in {lang}...')
            output_path = os.path.join(data_dir, lang, f'{corpus_name}_hard_negatives.jsonl')
            if os.path.exists(output_path):
                print(f'{corpus_name} has been processed!')
            else:
                data = load_miracl_datasets(lang=lang)
                data.to_json(output_path, lines=True, orient='records')
    elif corpus_name == 'vihealthqa':
        output_path = os.path.join(data_dir, 'vi', f'{corpus_name}.jsonl')
        print(f'Processing {corpus_name}...')
        if os.path.exists(output_path):
            print(f'{corpus_name} has been processed!')
        else:
            parse_beir_format(
            data_dir=data_dir,
            name='vihealthqa',
            output_path=output_path
        )
        if os.path.exists(output_path.replace('.jsonl', '_hard_negatives.jsonl')):
            print('vihealthqa hard negatives has been mined!')
        else:
            mine_hard_negatives(
                data_dir=data_dir,
                name='vihealthqa',
                data_path=output_path
            )
        
            
    


