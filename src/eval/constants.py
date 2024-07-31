MTEB_DS_TO_PROMPT = {
    "Classification": {
        "AmazonCounterfactualClassification": "Classify a given Amazon customer review text as either counterfactual or not-counterfactual.",
        "AmazonPolarityClassification": "Classify Amazon reviews as positive or negative.",
        "AmazonReviewsClassification": "Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.",
        "Banking77Classification": "Given a online banking query, find the corresponding intents.",
        "EmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the six emotions:anger, fear, joy, love, sadness, and surprise.",
        "ImdbClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MTOPDomainClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "ToxicConversationsClassification": "Classify the given comments as either toxic or not toxic.",
        "TweetSentimentExtractionClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
    },
    "Clustering": {
        'ArxivClusteringP2P': 'Identify the category of the following passages.',
        'ArxivClusteringS2S': 'Identify the category of the following passages.',
        'BiorxivClusteringP2P': 'Identify the category of the following passages.',
        'BiorxivClusteringS2S': 'Identify the category of the following passages.',
        'MedrxivClusteringP2P': 'Identify the category of the following passages.',
        'MedrxivClusteringS2S': 'Identify the category of the following passages.',
        'RedditClustering': 'Identify the category of the following passages.',
        'RedditClusteringP2P': 'Identify the category of the following passages.',
        'StackExchangeClustering': 'Identify the category of the following passages.',
        'StackExchangeClusteringP2P': 'Identify the category of the following passages.',
        'TwentyNewsgroupsClustering': 'Identify the category of the following passages.',
    },
    "PairClassification": {
        'SprintDuplicateQuestions': 'Retrieve semantically similar text.',
        'TwitterSemEval2015': 'Retrieve semantically similar text.',
        'TwitterURLCorpus': 'Retrieve semantically similar text.',
    },
    "Reranking": {
        'AskUbuntuDupQuestions':'Retrieve semantically similar text.',
        'MindSmallReranking': "Given a question, retrieve passages that answer the question.",
        'SciDocsRR': "Given a question, retrieve passages that answer the question.",
        'StackOverflowDupQuestions': 'Retrieve semantically similar text.',
    },
    'Retrieval': {
        'ArguAna': "Given a claim, retrieve documents that support or refute the claim.",
        'ClimateFEVER': "Given a claim, retrieve documents that support or refute the claim.",
        'CQADupstackTexRetrieval': 'Retrieve semantically similar text.',
        'DBPedia': "Given a question, retrieve passages that answer the question.",
        'FEVER': "Given a claim, retrieve documents that support or refute the claim.",
        'FiQA2018': "Given a question, retrieve passages that answer the question.",
        'HotpotQA': "Given a question, retrieve passages that answer the question.",
        'MSMARCO': "Given a question, retrieve passages that answer the question.",
        'NFCorpus': "Given a question, retrieve passages that answer the question.",
        'NQ': "Given a question, retrieve passages that answer the question.",
        'QuoraRetrieval': 'Retrieve semantically similar text.',
        'SCIDOCS': "Given a question, retrieve passages that answer the question.",
        'SciFact': "Given a claim, retrieve documents that support or refute the claim.",
        'Touche2020': "Given a question, retrieve passages that answer the question.",
        'TRECCOVID': "Given a claim, retrieve documents that support or refute the claim.",
    },
    'STS': {
        'STS12': "Retrieve semantically similar text.",
        'STS13': "Retrieve semantically similar text.",
        'STS14': "Retrieve semantically similar text.",
        'STS15': "Retrieve semantically similar text.",
        'STS16': "Retrieve semantically similar text.",
        'STS17': "Retrieve semantically similar text.",
        'STS22': "Retrieve semantically similar text.",
        'BIOSSES': "Retrieve semantically similar text.",
        'SICK-R': "Retrieve semantically similar text.",
        'STSBenchmark': "Retrieve semantically similar text.",
    },            
    'Summarization': {
        'SummEval': "Retrieve semantically similar text.",
    },
}


MULTILINGUAL_DS_TO_PROMPT = {
    'ar': {
        "TweetEmotionClassification": "Classify the emotion expressed in the given Twitter message into one of the following emotions: none, sympathy, anger, fear, joy, love, sadness, and surprise.",
        "ArEntail": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "MintakaRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "MLQARetrieval": "Given a question, retrieve passages that answer the question.",
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "STS17": "Retrieve semantically similar text.",
        'STS22.v2': "Retrieve semantically similar text.",
    },
    'en': {
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
    },
    'bn': {
        "BengaliDocumentClassification": 'Identify the category of the following passages.',
        "BengaliHateSpeechClassification": "Classify the following text into one of the following categories: Geopolitical, Personal, Political, Religious, or Gender abusive.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "IndicReviewsClusteringP2P": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "XNLIV2": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "WikipediaRerankingMultilingual": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "IndicQARetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "WikipediaRetrievalMultilingual": "Given a question, retrieve passages that answer the question.",
        "IndicCrosslingualSTS": "Retrieve semantically similar text.",
    },
    'de': {
        'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual.',
        "AmazonReviewsClassification": "Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.",
        "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MultilingualSentimentClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "TweetSentimentClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
        "TenKGnadClusteringP2P.v2": 'Identify the category of the following passages.',
        "TenKGnadClusteringS2S.v2": 'Identify the category of the following passages.',
        "BlurbsClusteringP2P.v2": 'Identify the category of the following passages.',
        "BlurbsClusteringS2S.v2": 'Identify the category of the following passages.',
        "MultiEURLEXMultilabelClassification": 'Identify the category of the following passages.',
        "RTE3": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "WikipediaRerankingMultilingual": "Given a question, retrieve passages that answer the question.",
        "GermanDPR": "Given a question, retrieve passages that answer the question.",
        "GermanGovServiceRetrieval": "Given a question, retrieve passages that answer the question.",
        "GermanQuAD-Retrieval": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MintakaRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "MIRACLRetrieval": "Given a question, retrieve passages that answer the question.",
        "WikipediaRetrievalMultilingual": "Given a question, retrieve passages that answer the question.",
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "XQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "STS17": "Retrieve semantically similar text.",
        'STS22': "Retrieve semantically similar text.",
        "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text.",
    },
    'es': {
        "AmazonReviewsClassification": "Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "MultilingualSentimentClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "TweetSentimentClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
        "SpanishNewsClassification": "Identify the category of the following passages.",
        "SpanishNewsClusteringP2P": "Identify the category of the following passages.",
        "MLSUMClusteringP2P.v2": "Identify the category of the following passages.",
        "MLSUMClusteringS2S.v2": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "MultiEURLEXMultilabelClassification": 'Identify the category of the following passages.',
        "PawsXPairClassification": "Retrieve semantically similar text.",
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MintakaRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "XQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "STS17": "Retrieve semantically similar text.",
        'STS22': "Retrieve semantically similar text.",
        "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text.",
    },
    'fa': {
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MultilingualSentimentClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "FarsTail": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "WikipediaRerankingMultilingual": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "NeuCLIR2023Retrieval": "Given a question, retrieve passages that answer the question.",
        "WikipediaRetrievalMultilingual": "Given a question, retrieve passages that answer the question.",
    },
    'fi': {
        "FinToxicityClassification": "Classify the given text into one of the following categories: identity attack, insult, obscene, severe toxicity, threat, toxicity",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MultilingualSentimentClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "SIB200Classification": "Identify the category of the following passages.",
        "OpusparcusPC": "Retrieve semantically similar text.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "WikipediaRerankingMultilingual": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "WikipediaRetrievalMultilingual": "Given a question, retrieve passages that answer the question.",
        "FinParaSTS": "Retrieve semantically similar text.",
    },
    'fr': {
        "AmazonReviewsClassification": "Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.",
        "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MasakhaNEWSClusteringP2P": "Identify the category of the following passages.",
        "TweetSentimentClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
        "SIB200Classification": "Identify the category of the following passages.",
        "FrenchBookReviews": "Classify the sentiment of a given book review as either positive, negative, or neutral.",
        "MasakhaNEWSClusteringS2S": "Identify the category of the following passages.",
        "MLSUMClusteringP2P.v2": "Identify the category of the following passages.",
        "MLSUMClusteringS2S.v2": "Identify the category of the following passages.",
        "AlloProfClusteringP2P.v2": "Identify the category of the following passages.",
        "AlloProfClusteringS2S.v2": "Identify the category of the following passages.",
        "HALClusteringS2S.v2": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "MultiEURLEXMultilabelClassification": 'Identify the category of the following passages.',
        "OpusparcusPC": "Retrieve semantically similar text.",
        "PawsXPairClassification": "Retrieve semantically similar text.",
        "RTE3": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "AlloprofReranking": "Given a question, retrieve passages that answer the question.",
        "SyntecReranking": "Given a question, retrieve passages that answer the question.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "AlloprofRetrieval": "Given a question, retrieve passages that answer the question.",
        "FQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MintakaRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "PublicHealthQA": "Given a question, retrieve passages that answer the question.",
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "STS17": "Retrieve semantically similar text.",
        "SICKFr": "Retrieve semantically similar text.",
        'STS22.v2': "Retrieve semantically similar text.",
        "STSBenchmarkMultilingualSTS": "Retrieve semantically similar text.",
        "SummEvalFr": "Retrieve semantically similar text.",
    },
    'hi': {
        "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "SentimentAnalysisHindi": "Classify the sentiment expressed in the given text as positive or negative.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "SIB200Classification": "Identify the category of the following passages.",
        "TweetSentimentClassification": "Classify the sentiment of a given tweet as either positive, negative, or neutral.",
        "IndicReviewsClusteringP2P": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "WikipediaRerankingMultilingual": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MintakaRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "MLQARetrieval": "Given a question, retrieve passages that answer the question.",
        "WikipediaRetrievalMultilingual": "Given a question, retrieve passages that answer the question.",
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "XQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "IndicCrosslingualSTS": "Retrieve semantically similar text.",
        "SemRel24STS": "Retrieve semantically similar text.",
    },
    'id': {
        "IndonesianMongabayConservationClassification": "Identify the category of the following passages.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "SIB200Classification": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "indonli": "Given a premise, retrieve a hypothesis that is entailed by the premise.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "SemRel24STS": "Retrieve semantically similar text.",
    },
    'ja': {
        'AmazonCounterfactualClassification': 'Classify a given Amazon customer review text as either counterfactual or not-counterfactual.',
        "AmazonReviewsClassification": "Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.",
        "WRIMEClassification": "Identify the category of the following passages.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "SIB200Classification": "Identify the category of the following passages.",
        "LivedoorNewsClustering.v2": "Identify the category of the following passages.",
        "MewsC16JaClustering": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "PawsXPairClassification": "Retrieve semantically similar text.",
        "VoyageMMarcoReranking": "Given a question, retrieve passages that answer the question.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "JaGovFaqsRetrieval": "Given a question, retrieve passages that answer the question.",
        "JaQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "NLPJournalAbsIntroRetrieval": "Given a question, retrieve passages that answer the question.",
        "NLPJournalTitleAbsRetrieval": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MintakaRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "JSICK": "Retrieve semantically similar text.",
        "JSTS": "Retrieve semantically similar text.",
    },
    'ko': {
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "KLUE-TC": "Identify the category of the following passages.",
        "KorFin": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "KorSarcasmClassification": "Classify the given text into one of the following categories: sarcasm or not sarcasm.",
        "SIB200Classification": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "KorHateSpeechMLClassification": "Identify the category of the following passages.",
        "PawsXPairClassification": "Retrieve semantically similar text.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "Ko-StrategyQA": "Given a question, retrieve passages that answer the question.",
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "PublicHealthQA": "Given a question, retrieve passages that answer the question.",
        "XPQARetrieval": "Given a question, retrieve passages that answer the question.",
        "KLUE-STS": "Retrieve semantically similar text.",
        "KorSTS": "Retrieve semantically similar text.",
        "STS17": "Retrieve semantically similar text.",
    },
    'ru': {
        "NeuCLIR2023Retrieval": "Given a question, retrieve passages that answer the question.",
        "GeoreviewClassification": "Identify the category of the following passages.",
        "GeoreviewClusteringP2P": "Identify the category of the following passages.",
        "HeadlineClassification": "Identify the category of the following passages.",
        "InappropriatenessClassification": "Identify the category of the following passages.",
        "KinopoiskClassification": "Identify the category of the following passages.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "RiaNewsRetrieval": "Given a question, retrieve passages that answer the question.",
        "RuBQRetrieval": "Given a question, retrieve passages that answer the question.",
        "RuReviewsClassification": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "RuSciBenchGRNTIClassification": "Identify the category of the following passages.",
        "RuSciBenchGRNTIClusteringP2P": "Identify the category of the following passages.",
        "RuSciBenchOECDClassification": "Identify the category of the following passages.",
        "RuSciBenchOECDClusteringP2P": "Identify the category of the following passages.",
        "TERRa": "Identify the category of the following passages.",
        "RuSTSBenchmarkSTS": "Retrieve semantically similar text.",
        'STS22': "Retrieve semantically similar text.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
    },
    'sw': {
        "AfriSentiClassification": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "MasakhaNEWSClassification": "Identify the category of the following passages.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "SwahiliNewsClassification": "Identify the category of the following passages.",
        "MasakhaNEWSClusteringP2P": "Identify the category of the following passages.",
        "MasakhaNEWSClusteringS2S": "Identify the category of the following passages.",
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
    },
    'te': {
        "IndicNLPNewsClassification": "Identify the category of the following passages.",
        "IndicSentimentClassification": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "SIB200Classification": "Identify the category of the following passages.",
        "TeluguAndhraJyotiNewsClassification": "Identify the category of the following passages.",
        "IndicReviewsClusteringP2P": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "IndicQARetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "IndicCrosslingualSTS": "Retrieve semantically similar text.",
        "SemRel24STS": "Retrieve semantically similar text.",
    },
    'th': {
        "MTOPIntentClassification": "Classify the intent of the given utterance in task-oriented conversation.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MultilingualSentimentClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "WisesightSentimentClassification": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "XQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
    },
    'vi': {
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MultilingualSentimentClassification": 'Classify the sentiment expressed in the given text as positive or negative.',
        "SIB200Classification": "Identify the category of the following passages.",
        "VieStudentFeedbackClassification": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "XNLI": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MLQARetrieval": "Given a question, retrieve passages that answer the question.",
        "PublicHealthQA": "Given a question, retrieve passages that answer the question.",
        "XQuADRetrieval": "Given a question, retrieve passages that answer the question.",
        "VieQuADRetrieval": "Given a question, retrieve passages that answer the question.",
    },
    'yo': {
        "AfriSentiClassification": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "MasakhaNEWSClassification": "Identify the category of the following passages.",
        "NaijaSenti": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "SIB200Classification": "Identify the category of the following passages.",
        "MasakhaNEWSClusteringP2P": "Identify the category of the following passages.",
        "MasakhaNEWSClusteringS2S": "Identify the category of the following passages.",
        "SIB200ClusteringS2S": "Identify the category of the following passages.",
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "BelebeleRetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
    },
    'zh': {
        "AmazonReviewsClassification": "Classify the given Amazon review into one of five rating categories: Poor, Fair, Good, Very good, Excellent.",
        "MLQARetrieval": "Given a question, retrieve passages that answer the question.",
        "MIRACLRetrieval": 'Given a question, retrieve passages that answer the question.',
        "MIRACLReranking": 'Given a question, retrieve passages that answer the question.',
        "IFlyTek": "Identify the category of the following passages.",
        "JDReview": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "MassiveIntentClassification": "Given a user utterance as query, find the user intents.",
        "MassiveScenarioClassification": "Given a user utterance as query, find the user scenarios.",
        "MultilingualSentiment": "Classify the sentiment expressed in the given text as positive, negative or neutral.",
        "OnlineShopping": "Classify the sentiment expressed in the given text as positive or negative.",
        "TNews": "Identify the category of the following passages.",
        "Waimai": "Classify the sentiment expressed in the given text as positive or negative.",
        "CLSClusteringP2P": "Identify the category of the following passages.",
        "CLSClusteringS2S": "Identify the category of the following passages.",
        "ThuNewsClusteringP2P": "Identify the category of the following passages.",
        "ThuNewsClusteringS2S": "Identify the category of the following passages.",
        "Cmnli": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "Ocnli": 'Given a premise, retrieve a hypothesis that is entailed by the premise.',
        "PawsXPairClassification": "Retrieve semantically similar text.",
        "MMarcoReranking": "Given a question, retrieve passages that answer the question.",
        "T2Reranking": "Given a question, retrieve passages that answer the question.",
        "CMedQAv2-reranking": "Given a question, retrieve passages that answer the question.",
        "CmedqaRetrieval": "Given a question, retrieve passages that answer the question.",
        "CovidRetrieval": "Given a question, retrieve passages that answer the question.",
        "DuRetrieval": "Given a question, retrieve passages that answer the question.",
        "EcomRetrieval": "Given a question, retrieve passages that answer the question.",
        "MedicalRetrieval": "Given a question, retrieve passages that answer the question.",
        "MMarcoRetrieval": "Given a question, retrieve passages that answer the question.",
        "T2Retrieval": "Given a question, retrieve passages that answer the question.",
        "VideoRetrieval": "Given a question, retrieve passages that answer the question.",
        "AFQMC": "Retrieve semantically similar text.",
        "ATEC": "Retrieve semantically similar text.",
        "BQ": "Retrieve semantically similar text.",
        "LCQMC": "Retrieve semantically similar text.",
        "PAWSX": "Retrieve semantically similar text.",
        "QBQTC": "Retrieve semantically similar text.",
        "STSB": "Retrieve semantically similar text.",
        'STS22': "Retrieve semantically similar text.",
    },
}

LANG_TO_CODES = {
    'ar': ['ara'],
    'bn': ['ben'],
    'de': ['deu'],
    'es': ['spa'],
    'en': ['eng'],
    'fa': ['fas'],
    'fi': ['fin'],
    'fr': ['fra'],
    'hi': ['hin'],
    'id': ['ind'],
    'ja': ['jpn'],
    'ko': ['kor'],
    'ru': ['rus'],
    'sw': ['swa'],
    'te': ['tel'],
    'th': ['tha'],
    'vi': ['vie'],
    'yo': ['yor'],
    'zh': ['cmn', 'cmo', 'zho'],
}

QUICK_EVAL = [
    # Classification
    "Banking77Classification",
    "EmotionClassification",
    # Clustering
    "MedrxivClusteringS2S",
    "TERRa",
    # PairClassification
    "TwitterSemEval2015",
    # Reranking
    "AskUbuntuDupQuestions",
    # Retrieval
    "ArguAna",
    "NFCorpus",
    "SciFact",
    'MintakaRetrieval',
    'BelebeleRetrieval',
    # STS
    "BIOSSES",
    "STSBenchmark",
    "STS22",
    # Summarization
    "SummEval",
]

