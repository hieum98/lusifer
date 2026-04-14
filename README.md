# *LUSIFER: Language Universal Space Integration for Enhanced Multilingual Embeddings with Large Language Models* (SIGIR 2025)

[![ArXiv](https://img.shields.io/badge/ArXiv-2025-fb1b1b.svg)](https://arxiv.org/abs/2501.00874)
[![HF Paper](https://img.shields.io/badge/HF%20Paper-2025-b31b1b.svg)](https://huggingface.co/papers/2501.00874)
[![HF Link](https://img.shields.io/badge/HF%20Model-LUSIFER-FFD21E.svg)](https://huggingface.co/Hieuman/LUSIFER)
[![DOI](https://img.shields.io/badge/DOI-10.1145/3726302.3730029-blue.svg)](https://doi.org/10.1145/3726302.3730029)
[![License](https://img.shields.io/badge/License-MIT-FD21E.svg)](LICENSE)

LUSIFER is framework for bridging the gap between multilingual understanding and task-specific text embeddings without relying on explicit multilingual supervision. It does this by combining a multilingual encoder (providing a universal language foundation) with an LLM-based embedding model (optimized for embedding tasks), connected through a minimal set of trainable parameters. LUSIFER also introduces two stages of training process: 1) Alignment Training and 2) Representation Fine-tuning to optimize the model for zero-shot multilingual embeddings.

<p align="center">
  <img src="https://github.com/hieum98/lusifer/blob/main/asserts/Model_overview.png" width="85%" alt="LUSIFER_figure1"/>
</p>

## Installation
To use LUSFIER, install evironment from ```environment.yaml``` (optional)
```bash
conda env create -f environment.yaml
```

After that, you can install our package from source by
```bash
pip install -e .
```

You also need to install the Flash-Attention before running the code because we use the Flash-Attention as the attention implementation in our model. You can install the Flash-Attention by running the following command:
```bash
pip install packaging
pip install ninja
pip install flash-attn --no-build-isolation
```

## Getting started
LUSIFER provides a thorough set of tools for training, evaluating, and using the model. The following sections provide a brief overview of how to use the model for training, evaluation, and inference.

### Preparing the model
LUSIFER model can be easily loaded using the `from_pretrained` method. The model can be loaded from the Hugging Face model hub by providing the model name or path to the model weights. The following code snippet demonstrates how to load the model from the Hugging Face model hub.

```python
from lusifer.models.lusifer import Lusifer

model = Lusifer.from_pretrained("Hieuman/LUSIFER")
```

### Inference
This model now returns the text embedding for any input in the form of `str` or `List[str]`. The model also can receive instruction alongside the sentence.

```python
import torch
from lusifer.models.lusifer import Lusifer

model = Lusifer.from_pretrained("Hieuman/LUSIFER")

model = model.to("cuda")

# Encoding queries using instructions
instruction =  "Given a web search query, retrieve relevant passages that answer the query:"
queries = [
    "how much protein should a female eat",
    "summit define",
]
q_reps = model.encode(sentences=queries)

# Encoding documents. Instruction are not required for documents
documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
]
d_reps = model.encode(sentences=documents)

# Compute cosine similarity
q_reps_norm = torch.nn.functional.normalize(torch.from_numpy(q_reps), p=2, dim=1)
d_reps_norm = torch.nn.functional.normalize(torch.from_numpy(d_reps), p=2, dim=1)
cos_sim = torch.mm(q_reps_norm, d_reps_norm.transpose(0, 1))

print(cos_sim)
```

## Training 

### Alignment Training
To train the model in the alignment stage, run the following command:
```bash
python -m src.main     \
    --config_file scripts/configs/aligment_training_reconstruction_and_completion.yaml     \
    --nodes 1     \
    --devices 4  
```
It will run the alignment training on 4 GPUs with both reconstruction and completion tasks with the configuration in the `scripts/configs/aligment_training_reconstruction_and_completion.yaml` file. For more details about the configuration file, please refer to the `scripts/configs/aligment_training_reconstruction_and_completion.yaml` file and the arguments in the `lusifer/args.py` file. 

We also provide the configuration file for the alignment training with the reconstruction task only in the `scripts/configs/alignment_training_reconstruction.yaml` file. We suggest using the reconstruction task only first to stabilize the training process before adding the completion task.

### Representation Fine-tuning
To train the model in the representation fine-tuning stage, run the following command:
```bash
python -m src.main     \
    --config_file scripts/configs/representation_fintuning_retrieval_data_only.yaml     \
    --nodes 1     \
    --devices 4  
```

We also provide the configuration file for the representation fine-tuning with both retrieval and non-retrieval data in the `scripts/configs/representation_finetuning_all.yaml` file. We suggest using the retrieval data only first to stabilize the training process before adding the non-retrieval data.

To be concise, we suggest the following training process: reconstruction task only -> reconstruction + completion task -> retrieval data only -> retrieval + non-retrieval data.

## Evaluation 
We propose a new benchmark for evaluating the model on the multilingual text embedding task. The benchmark includes 5 primary embedding tasks:  Classification, Clustering, Reranking, Retrieval, and Semantic Textual Similarity (STS) across 123 diverse datasets spanning 14 languages

<p align="center">
  <img src="https://github.com/hieum98/lusifer/blob/main/asserts/Benchmark.png" width="85%" alt="Benchmark"/>
</p>

We support to evaluate model on various datasets by intergrating [`mteb`](https://github.com/embeddings-benchmark/mteb) library. To evaluate the model, run the following command:
```bash
python -m lusifer.eval.eval \
    --model_name_or_path Hieuman/LUSIFER \
    --is_lusifer \
```

## Results
We provide the results of LUSIFER on the multilingual text embedding benchmark in the following table. The results are reported in terms of the average main metric across all tasks and datasets.

<p align="center">
  <img src="https://github.com/hieum98/lusifer/blob/main/asserts/Results.png" width="85%" alt="results"/>
</p>

## Citation
If you use LUSIFER in your research, please cite [our SIGIR 2025 paper](https://doi.org/10.1145/3726302.3730029):
```bibtex
@inproceedings{10.1145/3726302.3730029,
    author = {Man, Hieu and Ngo, Nghia Trung and Dac Lai, Viet and Rossi, Ryan A. and Dernoncourt, Franck and Huu Nguyen, Thien},
    title = {LUSIFER: Language Universal Space Integration for Enhanced Representation in Multilingual Text Embedding Models},
    year = {2025},
    isbn = {9798400715921},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3726302.3730029},
    doi = {10.1145/3726302.3730029},
    abstract = {Recent advancements in large language models (LLMs) based embedding models have established new state-of-the-art benchmarks for text embedding tasks, particularly in dense vector-based retrieval. However, these models predominantly focus on English, leaving multilingual embedding capabilities largely unexplored. To address this limitation, we present LUSIFER, a novel zero-shot approach that adapts LLM-based embedding models for multilingual tasks without requiring multilingual supervision. LUSIFER's architecture combines a multilingual encoder, serving as a language-universal learner, with an LLM-based embedding model optimized for embedding-specific tasks. These components are seamlessly integrated through a minimal set of trainable parameters that act as a connector, effectively transferring the multilingual encoder's language understanding capabilities to the specialized embedding model. Additionally, to comprehensively evaluate multilingual embedding performance, we introduce a new benchmark encompassing 5 primary embedding tasks, 123 diverse datasets, and coverage across 14 languages. Extensive experimental results demonstrate that LUSIFER significantly enhances the multilingual performance across various embedding tasks, particularly for medium and low-resource languages, without requiring explicit multilingual training data. The code and dataset for training are available at: https://github.com/hieum98/lusifer},
    booktitle = {Proceedings of the 48th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {1360–1370},
    numpages = {11},
    keywords = {large language models, multilingual benchmarks, multilingual text embedding, representation learning, zero-shot learning},
    location = {Padua, Italy},
    series = {SIGIR '25}
}

```

## Bugs or questions?
If you have any questions about the code, feel free to open an issue on the GitHub repository or send me an email at hieum@uoregon.edu.

