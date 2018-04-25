# Learning Word Embeddings Using Definitions

## Overview
This is a re-implementation of [Learning to Understand Phrases by Embedding the Dictionary](https://arxiv.org/abs/1504.00548) in PyTorch.

## Prerequirement
1. Input Data
	* Dictionary file: each line is formatted as `word <tab> definition sentence`. [[download]](https://www.dropbox.com/s/eriptqlofkxvx6x/glove.6B.100d.pk?dl=0)
	* Pretrained word embedding pickle file: a python dict (key: word, value: embedding list). [[download]](https://www.dropbox.com/s/bogsy2hwsqs6rud/felix-train.txt?dl=0)
	* A folder containing evaluation files for the reverse dictionary task (will remove from the main repo later). [[download]](https://www.dropbox.com/s/fltah0yneeyet2g/rvd.zip?dl=0)

2. Python 3.6
	* tqdm
	* PyTorch 0.31

## Pipeline
### Preprocess - tokenized dictionary file

```
cd scripts
```

Tokenize the original dictionary file. e.g. `a sweet fruit, with red (yellow) skin` to `a sweet fruit , with red ( yellow ) skin`

```
python tokenize_dict.py -f $PATH_DIC_FILE -o $PATH_OUT_FILE
```

Arguments:
* $PATH_DIC_FILE: path to the original dictionary file
* $PATH_OUT_FILE: path to the tokenized dictionary file

### Preprocess - generate training data

Pack training and evaluation data in a single pickle file. Just an intermediate input file to make it easier for the model to read.

```
python preprocess.py -r $PATH_RVD_DIR -e $PATH_EMBEDDING -d $PATH_DIC_FILE -o $PATH_DATA -v $VOCAB_SIZE --valid_ratio $VALID_RATIO
```

Arguments:
* $PATH_RVD_DIR: path to the reverse dictionary file folder
* $PATH_EMBEDDING: path to the embedding pickle file
* $PATH_DIC_FILE: path to the tokenized dictionary file (the output file of last step)
* $PATH_DATA: path to the output pickle file
* $VOCAB_SIZE (default 30000): We only pick the most frequent $VOCAB_SIZE words as input. Other words will be replaced by `<unk>`
* $VALID_RATIO (default 0.05): ration of the validation set in the entire training set

Output content:
```
{
    'args': vars(args),
    'dic_embed': dic_embed,
    'def_embed': def_embed,
    'dic_word2ix': dic_word2ix,
    'def_word2ix': def_word2ix,
    'train_pairs': train_pairs,
    'valid_pairs': valid_pairs,
    'rvd_candidates': rvd_candidates,
    'rvd_pairs_seen': rvd_pairs_seen,
    'rvd_pairs_unseen': rvd_pairs_unseen,
    'rvd_pairs_concept': rvd_pairs_concept,
}
```

### Training

```
cd src
python train.py -g $GPU --data $PATH_DATA -c $PATH_CHECKPOINT
```

Arguments:
* $GPU: which gpu to use. `-1` means train on CPU.
* $PATH_DATA: path to the input data file (the output file of last step)
* $PATH_CHECKPOINT (optional): path to the checkpoint. You can skip this argument and the model will store the checkpoint to the default path.

### Inference

Please refer to `scripts/case.ipynb` for now on how to initialize the model and do the inference.


