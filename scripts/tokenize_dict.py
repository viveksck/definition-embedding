import argparse
from tqdm import tqdm
from nltk.tokenize import word_tokenize

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dic_file', type=str, required=True)
    parser.add_argument('-o', '--out_file', type=str, required=True)
    args = parser.parse_args()

    with open(args.dic_file) as rf:
        lines = rf.read().splitlines()

    with open(args.out_file, 'w') as wf:
        for line in tqdm(lines, ncols=10):
            try:
                word, sent = line.split('\t')
            except ValueError:
                continue
            def_words = word_tokenize(sent)
            tokenized_sent = ' '.join(def_words)
            wf.write('{}\t{}\n'.format(word, tokenized_sent))