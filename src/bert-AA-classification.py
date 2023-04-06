import os
import argparse
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from bert_experiments.bertutils import *
import pathlib

from utils.evaluation import evaluation_attribution, evaluation_verification
from utils.result_manager import AttributionResult, check_if_already_performed

"""
This scripts takes a pre-trained model (fine-tuned or not) and performs authorship attribution.
"""




def main(args):
    assert torch.cuda.is_available(), 'cuda is not available'

    csv = AttributionResult(args.logfile)

    checkpoint = join(args.checkpoint, 'best_checkpoint')

    Xtr, ytr, Xte, yte = load_dataset(args)
    Xte = instances_as_torch_dataset(Xte)
    num_labels = len(np.unique(yte))

    with torch.no_grad():
        print('loading', checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()
        yte_ = classify(model, tokenizer, Xte)

    # evaluation
    acc_attr, f1_attr_macro, f1_attr_micro = evaluation_attribution(yte, yte_)
    acc_verif, f1_verif_macro, f1_verif_micro = evaluation_verification(yte, yte_)

    # output
    num_el = Xtr.shape[0]
    csv.append(args.dataset, args.seed, args.n_authors, args.docs_by_author, args.modelname, num_el,
               -1, -1, -1,
               acc_attr, f1_attr_macro, f1_attr_micro, f1_verif_micro, f1_verif_macro)
    csv.close()



if __name__ == '__main__':

    available_datasets = {'imdb62', 'pan2011', 'victorian', 'arxiv'}

    parser = argparse.ArgumentParser(description='Performs authorship attribution using a pre-trained model')
    parser.add_argument('dataset', type=str, metavar='STR',
                        help=f'Name of the dataset to run experiments on (valid ones are {available_datasets})')
    parser.add_argument('modelname', metavar='NAME', type=str,
                        help="Name given to the model for the result log")
    parser.add_argument('checkpoint', metavar='NAME', type=str,
                        help="Path to a pre-trained fine-tuned model, or name of a Huggingface's pre-trained model")
    parser.add_argument('--n_authors', type=int, default=-1, metavar='N', help='Number of authors to extract')
    parser.add_argument('--docs_by_author', type=int, default=-1, metavar='N',
                        help='Number of texts by author to extract')
    parser.add_argument('--logfile', type=str, default='../log/results.csv', help='File csv with results')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f'Pretrained model {args.checkpoint} not found')

    assert not check_if_already_performed(
        args.logfile, args.dataset, args.seed, args.n_authors, args.docs_by_author, args.modelname
    ), 'experiment already performed'

    parentdir = pathlib.Path(args.logfile).parent
    if parentdir:
        os.makedirs(parentdir, exist_ok=True)

    main(args)