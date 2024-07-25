import os
import datasets

from src.data_modules.constants import *


if __name__=="__main__":
    lang_to_data = {
            'en': EN,
            'ar': AR,
            'bn': BN,
            'de': DE,
            'es': ES,
            'fa': FA,
            'fi': FI,
            'fr': FR,
            'hi': HI,
            'id': ID,
            'ja': JA,
            'ko': KO,
            'ru': RU,
            'sw': SW,
            'te': TE,
            'th': TH,
            'vi': VI,
            'yo': YO,
            'zh': ZH,
        }
    
    max_num_worker_suggest = 1
    try:
        max_num_worker_suggest = len(os.sched_getaffinity(0))
    except Exception:
        pass
    num_workers = min(40, max_num_worker_suggest)
    
    data_names = []
    for l in lang_to_data.keys():
        data_names.extend(lang_to_data[l])

    for n in data_names:
        # print("Processing data_name:", n)
        data_path = DATA[n]['data_path']
        # try:
        #     data = datasets.load_dataset('json', data_files=data_path, split='train')
        #     if len(data) < 100:
        #         print("Skipping data_name:", n, "as it has less than 100 examples")
        #     continue
        # except:
        #     print("Error loading data_name:", n)
        #     continue
        # filter out data with no positive examples
        print("Original number of examples:", len(data))
        data = data.filter(lambda x: len(x['positive']) > 0, num_proc=num_workers)
        print("Number of examples after removing those with no positive examples:", len(data))
        # filter out data with no negative examples
        data = data.filter(lambda x: len(x['negative']) > 0, num_proc=num_workers)
        print("Number of examples after removing those with no negative examples:", len(data))
        # filter out data with too short queries
        if len(data) > 1000:
            data = data.filter(lambda x: len(x['query'].split()) > 3, num_proc=num_workers)
            print("Number of examples after removing those with too short queries:", len(data))

        # save the filtered data
        data.to_json(DATA[n]['data_path'], orient='records', lines=True, force_ascii=False)


