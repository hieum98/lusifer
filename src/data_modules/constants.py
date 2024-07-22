DATA = {
    'Arguana': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/arguana_hard_negatives.jsonl',
        'instruction': 'Given a claim, find documents that refute the claim.'
    },
    'CQADupStack': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/cqadupstack_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve other questions from Stackexchange that are duplicates to the given question.'
    },
    'DBPedia': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/dbpedia-entity_hard_negatives.jsonl',
        'instruction': 'Given a query, retrieve relevant entity descriptions from DBPedia.'
    },
    'Fever': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/fever_hard_negatives.jsonl',
        'instruction': 'Given a claim, retrieve documents that support or refute the claim.'
    },
    'FiQA': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/fiqa_hard_negatives.jsonl',
        'instruction': 'Given a financial question, retrieve user replies that best answer the question.'
    },
    'HotpotQA': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/hotpotqa_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'MSMarco': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/msmarco_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'NFCorpus': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/nfcorpus_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'NQ': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/nq_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'Quora': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/quora_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve questions that are semantically equivalent to the given question.'
    },
    'SciDocs': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/scidocs_hard_negatives.jsonl',
        'instruction': 'Given a scientific paper title, retrieve paper abstracts that are cited by the given paper.'
    },
    'SciFact': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/scifact_hard_negatives.jsonl',
        'instruction': 'Given a scientific claim, retrieve documents that support or refute the claim.'
    },
    'TREC-COVID': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/trec-covid_hard_negatives.jsonl',
        'instruction': 'Given a query on COVID-19, retrieve documents that answer the query.'
    },
    'Touche2020': {
        'data_path': '/home/hieum/uonlp/lusifer/data/beir/webis-touche2020_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    '20NewsGroups': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/20newsgroups_hard_negatives.jsonl',
        'instruction': 'Identify the topic or theme of the given news articles.'
    },
    'Amazon-Counterfactual': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/amazon_counterfactual_hard_negatives.jsonl',
        'instruction': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual.'
    },
    'Amazon-Review': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/amazon_review_hard_negatives.jsonl',
        'instruction': 'Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.'
    },
    'ArxivP2P': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/arxivP2P_hard_negatives.jsonl',
        'instruction': 'Identify the category of papers based on the titles and abstracts.'
    },
    'ArxivS2S': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/arxivS2S_hard_negatives.jsonl',
        'instruction': 'Identify the category of papers based on the titles.'
    },
    'Banking77': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/banking77_hard_negatives.jsonl',
        'instruction': 'Given a online banking query, find the corresponding intents.'
    },
    'biorxivP2P': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/biorxivP2P_hard_negatives.jsonl',
        'instruction': 'Identify the category of papers based on the titles and abstracts.'
    },
    'biorxivS2S': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/biorxivS2S_hard_negatives.jsonl',
        'instruction': 'Identify the category of papers based on the titles.'
    },
    'Emotion': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/emotion_hard_negatives.jsonl',
        'instruction': 'Classify the emotion expressed in the given Twitter message into one of the six emotions:anger, fear, joy, love, sadness, and surprise.'
    },
    'imdb': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/imdb_hard_negatives.jsonl',
        'instruction': 'Classify the sentiment expressed in the given movie review text as positive or negative.'
    },
    'medrxivP2P': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/medrxivP2P_hard_negatives.jsonl',
        'instruction': 'Identify the category of papers based on the titles and abstracts.'
    },
    'medrxivS2S': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/medrxivS2S_hard_negatives.jsonl',
        'instruction': 'Identify the category of papers based on the titles.'
    },
    'en_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'en_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'MTOP-Intent': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/mtop_intent_hard_negatives.jsonl',
        'instruction': 'Classify the intent of the given utterance in task-oriented conversation.'
    },
    'PQA': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/pqa_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'PubMedQA': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/pubmedqa_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'Reddit-clustering': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/reddit_clustering_hard_negatives.jsonl',
        'instruction': 'Identify the topic or theme of Reddit posts based on the titles and posts.'
    },
    'snli': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/snli_hard_negatives.jsonl',
        'instruction': 'Given a premise, retrieve a hypothesis that is entailed by the premise.'
    },
    'SQuAD': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/squad_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'StackExchange-clustering': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/stackexchange_clustering_hard_negatives.jsonl',
        'instruction': 'Identify the topic or theme of StackExchange posts based on the given paragraphs.'
    },
    'STS': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/sts_hard_negatives.jsonl',
        'instruction': 'Retrieve semantically similar text.'
    },
    'Toxic-conversation': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/toxic_conversations_hard_negatives.jsonl',
        'instruction': 'Classify the given comments as either toxic or not toxic.'
    },
    'Tweet-Sentiment': {
        'data_path': '/home/hieum/uonlp/lusifer/data/en/tweet_sentiments_hard_negatives.jsonl',
        'instruction': 'Classify the sentiment of a given tweet as either positive, negative, or neutral.'
    },

    'ar_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ar/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'ar_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ar/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'bn_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/bn/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'bn_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/bn/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'de_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/de/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'es_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/es/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'fa_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/fa/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'fi_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/fi/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'fi_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/fi/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'fr_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/fr/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'hi_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/hi/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'id_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/id/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'id_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/id/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'ja_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ja/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'ja_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ja/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'ko_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ko/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'ko_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ko/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'ru_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ru/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'ru_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/ru/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'sw_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/sw/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'sw_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/sw/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'te_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/te/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'te_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/te/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'th_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/th/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
    'th_MrTidy': {
        'data_path': '/home/hieum/uonlp/lusifer/data/th/mr-tidy_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'vi-HealthQA': {
        'data_path': '/home/hieum/uonlp/lusifer/data/vi/vihealthqa_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'yo_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/yo/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },

    'zh_MIRACL': {
        'data_path': '/home/hieum/uonlp/lusifer/data/zh/miracl_hard_negatives.jsonl',
        'instruction': 'Given a question, retrieve passages that answer the question.'
    },
}

EN = [
    'Arguana', 'CQADupStack', 'DBPedia', 'Fever', 
    'FiQA', 'HotpotQA', 'MSMarco', 'NFCorpus', 
    'NQ', 'Quora', 'SciDocs', 'SciFact', 'TREC-COVID', 
    'Touche2020', '20NewsGroups', 'Amazon-Counterfactual',
    'Amazon-Review', 'ArxivP2P', 'ArxivS2S', 'Banking77',
    'biorxivP2P', 'biorxivS2S', 'Emotion', 'imdb',
    'medrxivP2P', 'medrxivS2S', 'en_MIRACL', 'en_MrTidy',
    'MTOP-Intent', 'PQA', 'PubMedQA', 'Reddit-clustering',
    'snli', 'SQuAD', 'StackExchange-clustering', 'STS',
    'Toxic-conversation', 'Tweet-Sentiment'
    ]

AR = ['ar_MIRACL', 'ar_MrTidy']

BN = ['bn_MIRACL', 'bn_MrTidy']

DE = ['de_MIRACL']

ES = ['es_MIRACL']

FA = ['fa_MIRACL']

FI = ['fi_MIRACL', 'fi_MrTidy']

FR = ['fr_MIRACL']

HI = ['hi_MIRACL']

ID = ['id_MIRACL', 'id_MrTidy']

JA = ['ja_MIRACL', 'ja_MrTidy']

KO = ['ko_MIRACL', 'ko_MrTidy']

RU = ['ru_MIRACL', 'ru_MrTidy']

SW = ['sw_MIRACL', 'sw_MrTidy']

TE = ['te_MIRACL', 'te_MrTidy']

TH = ['th_MIRACL', 'th_MrTidy']

VI = ['vi-HealthQA']

YO = ['yo_MIRACL']

ZH = ['zh_MIRACL']
