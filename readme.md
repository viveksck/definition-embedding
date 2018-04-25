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
Tokenize the original dictionary file. e.g. `a sweet fruit, with red (yellow) skin` to `a sweet fruit , with red ( yellow ) skin`
***Arguments***:
	*  $PATH_DIC_FILE: path to the original dictionary file
	*  $PATH_OUT_FILE: path to the tokenized dictionary file
```
cd scripts
python tokenize_dict.py -f $PATH_DIC_FILE -o $PATH_OUT_FILE
```