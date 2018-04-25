# Learning Word Embeddings Using Definitions

## Overview


## Prerequirement
1. Input Data
	* Pretrained word embedding pickle file: a python dict (key: word, value: embedding list). [download](https://www.dropbox.com/s/bogsy2hwsqs6rud/felix-train.txt?dl=0)
	* Dictionary file: each line is formatted as `word <tab> definition sentence`. [download](https://www.dropbox.com/s/eriptqlofkxvx6x/glove.6B.100d.pk?dl=0)

2. Python environment
	* PyTorch 0.31
	* tqdm

## Pipeline
### 1. Preprocess - tokenized dictionary file

Tokenize the original dictionary file. e.g. `a sweet fruit, with red (yellow) skin` to `a sweet fruit , with red ( yellow ) skin`

***Arguments***:
	*  $PATH_DIC_FILE: path to the original dictionary file
	*  $PATH_OUT_FILE: path to the tokenized dictionary file

```
cd scripts
python tokenize_dict.py -f $PATH_DIC_FILE -o $PATH_OUT_FILE
```

2. Preprocess - 