#!/bin/bash

#SBATCH --nodes=1              # This needs to match Fabric(num_nodes=...)
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu-80gb|h100|gpu-40gb
#SBATCH --mem=100G
#SBATCH --cpus-per-task=5
#SBATCH --job-name=lusifer-eval
#SBATCH --partition=preempt,gpulong,cisds
#SBATCH --account=uonlp
#SBATCH --output=/home/hieum/uonlp/lusifer/lusifer-eval-%j.log
#SBATCH --error=/home/hieum/uonlp/lusifer/lusifer-eval-%j.log
#SBATCH --array=0-200

# Activate conda environment
source /home/hieum/.bashrc
conda activate lusifer
cd /home/hieum/uonlp/lusifer


DATA=(
'SpanishNewsClusteringP2P' 
'CMedQAv2-reranking'
'WikipediaRetrievalMultilingual'
'ArEntail'
'LivedoorNewsClustering.v2'
'SIB200Classification'
'WikipediaRerankingMultilingual'
'FQuADRetrieval'
'XQuADRetrieval'
'Ko-StrategyQA' 
'SyntecReranking'
'RuSciBenchOECDClusteringP2P'
'PublicHealthQA'
'Ocnli'
'SwahiliNewsClassification'
'T2Reranking'
'JSTS'
'VieStudentFeedbackClassification'
'STSB'
'MLSUMClusteringP2P.v2'
'MMarcoRetrieval'
'SICKFr'
'LCQMC'
'DuRetrieval'
'STS22'
'IndonesianMongabayConservationClassification'
'MMarcoReranking'
'IFlyTek'
'ThuNewsClusteringS2S'
'MewsC16JaClustering'
'MultilingualSentimentClassification'
'STSBenchmarkMultilingualSTS'
'TenKGnadClusteringS2S.v2'
'PAWSX'
'NLPJournalTitleAbsRetrieval'
'RuSTSBenchmarkSTS'
'MassiveIntentClassification'
'KLUE-TC'
'MLSUMClusteringS2S.v2'
'JSICK'
'IndicCrosslingualSTS'
'MasakhaNEWSClassification'
'IndicQARetrieval'
'HALClusteringS2S.v2'
'SemRel24STS'
'MultilingualSentiment'
'OpusparcusPC'
'JDReview'
'MassiveScenarioClassification'
'RiaNewsRetrieval'
'SpanishNewsClassification'
'FrenchBookReviews'
'KorSarcasmClassification'
'STS22.v2'
'TenKGnadClusteringP2P.v2'
'BengaliDocumentClassification'
'CmedqaRetrieval'
'FinParaSTS'
'RuSciBenchOECDClassification'
'GeoreviewClusteringP2P'
'PawsXPairClassification'
'VieQuADRetrieval'
'InappropriatenessClassification'
'MTOPIntentClassification'
'AlloprofReranking'
'AlloProfClusteringP2P.v2'
'HeadlineClassification'
'TERRa'
'MedicalRetrieval'
'IndicSentimentClassification'
'XNLIV2'
'MultiEURLEXMultilabelClassification'
'SentimentAnalysisHindi'
'BlurbsClusteringS2S.v2'
'XNLI'
'AlloProfClusteringS2S.v2'
'IndicReviewsClusteringP2P'
'JaQuADRetrieval'
'WRIMEClassification'
'ThuNewsClusteringP2P'
'MasakhaNEWSClusteringS2S'
'BQ'
'TweetSentimentClassification'
'CLSClusteringP2P'
'TweetEmotionClassification'
'TNews'
'JaGovFaqsRetrieval'
'RuReviewsClassification'
'SummEvalFr'
'GermanQuAD-Retrieval'
'NeuCLIR2023Retrieval'
'NLPJournalAbsIntroRetrieval'
'MIRACLReranking'
'GermanDPR'
'KorSTS'
'EcomRetrieval'
'NaijaSenti'
'XPQARetrieval'
'GeoreviewClassification'
'IndicNLPNewsClassification'
'TeluguAndhraJyotiNewsClassification'
'AmazonReviewsClassification'
'T2Retrieval'
'SIB200ClusteringS2S'
'indonli'
'VoyageMMarcoReranking'
'CLSClusteringS2S'
'STS17'
'AlloprofRetrieval'
'KLUE-STS'
'FarsTail'
'BengaliHateSpeechClassification'
'MasakhaNEWSClusteringP2P'
'RuSciBenchGRNTIClusteringP2P'
'WisesightSentimentClassification'
'KorHateSpeechMLClassification'
'RTE3'
'AFQMC'
'FinToxicityClassification'
'CovidRetrieval'
'AmazonCounterfactualClassification'
'AfriSentiClassification'
'OnlineShopping'
'RuSciBenchGRNTIClassification'
'MintakaRetrieval'
'MIRACLRetrieval'
'ATEC'
'MLQARetrieval'
'BlurbsClusteringP2P.v2'
'GermanGovServiceRetrieval'
'KinopoiskClassification'
'RuBQRetrieval'
'VideoRetrieval'
'BelebeleRetrieval'
'Waimai'

"AmazonCounterfactualClassification"
"AmazonPolarityClassification"
"AmazonReviewsClassification"
"Banking77Classification"
"EmotionClassification"
"ImdbClassification"
"MassiveIntentClassification"
"MassiveScenarioClassification"
"MTOPDomainClassification"
"MTOPIntentClassification"
"ToxicConversationsClassification"
"TweetSentimentExtractionClassification"
"SprintDuplicateQuestions"
"TwitterSemEval2015"
"TwitterURLCorpus"
"ArxivClusteringP2P"
"ArxivClusteringS2S"
"BiorxivClusteringP2P"
"BiorxivClusteringS2S"
"MedrxivClusteringP2P"
"MedrxivClusteringS2S"
"RedditClustering"
"RedditClusteringP2P"
"StackExchangeClustering"
"StackExchangeClusteringP2P"
"TwentyNewsgroupsClustering"
"AskUbuntuDupQuestions"
"MindSmallReranking"
"SciDocsRR"
"StackOverflowDupQuestions"
# "MIRACLReranking"
"ArguAna"
"ClimateFEVER"
"CQADupstackTexRetrieval"
"DBPedia"
"FEVER"
"FiQA2018"
"HotpotQA"
"MSMARCO"
"NFCorpus"
"NQ"
"QuoraRetrieval"
"SCIDOCS"
"SciFact"
"Touche2020"
"TRECCOVID"
# "MIRACLRetrieval"
"STS12"
"STS13"
"STS14"
"STS15"
"STS16"
"STS17"
"STS22"
"BIOSSES"
"SICK-R"
"STSBenchmark"
"SummEval"
)

LANGS=(
'ar'
'en'
'bn'
'de'
'es'
'fa'
'fi'
'fr'
'hi'
'id'
'ja'
'ko'
'ru'
'sw'
'te'
'vi'
'yo'
'zh'
)
# LANG=${LANGS[$SLURM_ARRAY_TASK_ID]}
# DATA_NAME='MIRACLRetrieval'
# echo "Evaluating $LANG"
# # if LANG is none, then exit
# if [ -z "$LANG" ]; then
#     echo "No lang to evaluate"
#     rm -rf /home/hieum/uonlp/lusifer/lusifer-eval-$SLURM_JOB_ID.log
#     exit 1
# fi

LANG='ar en bn de es fa fi fr hi id ja ko ru sw te vi yo zh'
DATA_NAME=${DATA[$SLURM_ARRAY_TASK_ID]}
echo "Evaluating $DATA_NAME"
# if DATA_NAME is none, then exit
if [ -z "$DATA_NAME" ]; then
    echo "No data to evaluate"
    rm -rf /home/hieum/uonlp/lusifer/lusifer-eval-$SLURM_JOB_ID.log
    exit 1
fi


export HF_HOME="/home/hieum/uonlp/hf_data_mteb"
# 
python -m lusifer.eval.eval \
    --model_name_or_path sentence-transformers/sentence-t5-xxl \
    --output_folder st5 \
    --batch_size 512 \
    --max_length 512 \
    --dataname $DATA_NAME \
    --langs $LANG  

# Check if the job is done
if [ $? -eq 0 ]; then
    echo "Job is done"
    rm -rf /home/hieum/uonlp/lusifer/lusifer-eval-$SLURM_JOB_ID.log
else
    echo "Job failed"
fi



