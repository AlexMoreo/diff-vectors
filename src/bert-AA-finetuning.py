import argparse
import csv
import importlib
import os
import sys
import datasets
import numpy as np
import pandas as pd
import torch.cuda
from datasets import Dataset, DatasetDict
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import Trainer
from transformers import TrainingArguments

from bert_experiments.bertutils import clean_checkpoints, load_dataset



"""
This script fine-tunes a pre-trained language model on a given textual training set.
The training goes for a maximum of 50 epochs, but stores the model parameters of the best performing epoch according
to the validation loss in a held-out val split of 20% documents (stratified).
"""


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    return {
        'macro-f1': f1_score(labels, preds, average='macro'),
        'micro-f1': f1_score(labels, preds, average='micro'),
    }


def main(args):
    assert torch.cuda.is_available(), 'cuda is not available'

    checkpoint = args.checkpoint
    modelout = args.modelname
    if modelout is None:
        modelout = checkpoint+'-finetuned'

    Xtr, ytr, Xte, yte = load_dataset(args)
    Xtr, Xva, ytr, yva = train_test_split(Xtr, ytr, stratify=ytr, test_size=0.2, random_state=1)
    num_labels = len(pd.unique(ytr))
    print('Num classes =', num_labels)

    df_tr = pd.DataFrame({'text': Xtr, 'label':ytr})
    df_va = pd.DataFrame({'text': Xva, 'label': yva})
    features = datasets.Features({'label': datasets.Value('int32'), 'text': datasets.Value('string')})
    train = Dataset.from_pandas(df=df_tr, split='train', features=features)
    validation = Dataset.from_pandas(df=df_va, split='validation', features=features)

    dataset = DatasetDict({
        'train': train,
        'validation': validation
    })

    # tokenize the dataset
    def tokenize_function(example):
        tokens = tokenizer(example['text'], padding='max_length', truncation=True, max_length=256)
        return tokens

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels).cuda()

    # fine-tuning
    training_args = TrainingArguments(
        modelout,
        learning_rate=1e-5,
        num_train_epochs=50,
        weight_decay=0.01,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        load_best_model_at_end=True
    )
    trainer = Trainer(
        model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        data_collator=DataCollatorWithPadding(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    trainer.train()

    clean_checkpoints(modelout, score='eval_macro-f1', higher_is_better=True, verbose=True)


if __name__ == '__main__':

    available_datasets = {'imdb62', 'pan2011', 'victorian', 'arxiv'}

    parser = argparse.ArgumentParser(description='Pre-trained Transformer finetuning')
    parser.add_argument('dataset', type=str, metavar='STR',
                        help=f'Name of the dataset to run experiments on (valid ones are {available_datasets})')
    parser.add_argument('--checkpoint', metavar='NAME', type=str, default='roberta-base',
                        help="Name of a Huggingface's pre-trained model")
    parser.add_argument('--modelname', metavar='NAME', type=str, default=None,
                        help="The name of the folder where the checkpoints will be dumped. "
                             "Default is None, meaning args.checkpoint+'-finetuned'")
    parser.add_argument('--n_authors', type=int, default=-1, metavar='N',
                        help='Number of authors to extract')
    parser.add_argument('--docs_by_author', type=int, default=-1, metavar='N',
                        help='Number of texts by author to extract')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    main(args)