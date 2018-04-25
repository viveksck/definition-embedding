# Learning Word Embeddings Using Definitions

## Overview


## Prerequirement
1. Input Data
* Pretrained word embedding pickle file ($PATH_EMBEDDING): a python dict (key: word, value: embedding list). e.g. `{'apple': [0.1, 0.2, 0.3]}`
* Dictionary file ($PATH_DIC_FILE): each line is formatted as `word <tab> definition sentence`. e.g. `apple<tab>a sweet fruit with red or yellow skin`

2. Python environment
* PyTorch 0.31
* tqdm

## Pipeline

1. Preprocess - tokenized dictionary file