import os
import argparse
import pathlib

import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from bert_experiments.bertutils import *
import pickle
from scipy.sparse import csr_matrix

"""
This scripts takes a pre-trained model (fine-tuned or not) and generates a version of any dataset (victorian, pan2011,
imdb62, or arxiv) so that every document is represented as a numeric vector corresponding to the CLS-embedding.
Datasets are pickled so that these can be used as any other dataset for the rest of the python scripts.
"""

def main(args):
    assert torch.cuda.is_available(), 'cuda is not available'

    checkpoint = join(args.checkpoint, 'best_checkpoint')

    Xtr, ytr, Xte, yte = load_dataset(args)
    Xtr = instances_as_torch_dataset(Xtr)
    Xte = instances_as_torch_dataset(Xte)
    num_labels = len(np.unique(yte))

    with torch.no_grad():
        print('loading', checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()
        Xtr = doc_embeddings(model, tokenizer, Xtr)
        Xte = doc_embeddings(model, tokenizer, Xte)

    Xtr = csr_matrix(Xtr)  # for compatibility with other pickles
    Xte = csr_matrix(Xte)  # for compatibility with other pickles

    print('Pickling dataset')
    print(f'Xtr.shape={Xtr.shape}')
    print(f'Xte.shape={Xte.shape}')
    print(f'Authors={len(np.unique(ytr))}')

    pickle.dump((Xtr, ytr, Xte, yte, 0), open(args.pickle, 'wb'), pickle.HIGHEST_PROTOCOL)

    print('[Done]')


if __name__ == '__main__':

    available_datasets = {'imdb62', 'pan2011', 'victorian', 'arxiv'}

    parser = argparse.ArgumentParser(description='Extract dense features for any dataset using a pre-trained model')
    parser.add_argument('dataset', type=str, metavar='STR',
                        help=f'Name of the dataset to run experiments on (valid ones are {available_datasets})')
    parser.add_argument('checkpoint', metavar='NAME', type=str,
                        help="Path to a pre-trained fine-tuned model, or name of a Huggingface's pre-trained model")
    parser.add_argument('pickle', metavar='NAME', type=str,
                        help="Path with the new name of the dataset")
    parser.add_argument('--n_authors', type=int, default=-1, metavar='N', help='Number of authors to extract')
    parser.add_argument('--docs_by_author', type=int, default=-1, metavar='N',
                        help='Number of texts by author to extract')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Pretrained model {args.checkpoint} not found')

    parentdir = pathlib.Path(args.pickle).parent
    if parentdir:
        os.makedirs(parentdir, exist_ok=True)

    main(args)