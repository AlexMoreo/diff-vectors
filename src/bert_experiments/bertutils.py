import datasets
from datasets import Dataset
import pandas as pd
import numpy as np
import json
from glob import glob
from os.path import join
import os
import shutil
import importlib
from tqdm import tqdm


def load_dataset(opt):
    dataset_loader = getattr(importlib.import_module(f'data.fetch_{opt.dataset.lower()}'), opt.dataset.title())
    dataset = dataset_loader(n_authors=opt.n_authors, docs_by_author=opt.docs_by_author, n_open_set_authors=0,
                             random_state=opt.seed)

    Xtr, ytr = dataset.train.data, dataset.train.target
    Xte, yte = dataset.test.data, dataset.test.target
    return Xtr, ytr, Xte, yte


def tokenize_function(tokenizer, example):
    tokens = tokenizer(example['text'], padding='max_length', truncation=True, max_length=256, return_tensors='pt')
    return {
        'input_ids': tokens.input_ids.cuda(),
        'attention_mask': tokens.attention_mask.cuda()
    }


# def save_samples_as_txt(vectors, labels, path):
#     cols = list(np.arange(vectors.shape[1]))
#     if labels is not None:
#         labels = labels.values
#         cols = ['label']+cols
#         vectors = np.hstack([labels, vectors])
#     df = pd.DataFrame(vectors, columns=cols)
#     if labels is not None:
#         df["label"] = pd.to_numeric(df["label"], downcast='integer')
#     df.to_csv(path, index=False)


def instances_as_torch_dataset(instances):
    features = datasets.Features({'text': datasets.Value('string')})
    df = pd.DataFrame({'text': instances})
    return Dataset.from_pandas(df=df, features=features)


def doc_embeddings(model, tokenizer, instances, batch_size=50):
    ndocs = len(instances)

    doc_embeddings = []
    for batch_id in tqdm(range(0, ndocs, batch_size), total=ndocs//batch_size, desc='projecting'):
        batch_instances = instances[batch_id:batch_id + batch_size]
        tokenized_dataset = tokenize_function(tokenizer, batch_instances)
        out = model(**tokenized_dataset, output_hidden_states=True)
        cls_embedding = out.hidden_states[-1][:,0,:]  # the last hidden state is (-1), the cls is in (0)
        doc_embeddings.append(cls_embedding.cpu().numpy())

    return np.concatenate(doc_embeddings)


def classify(model, tokenizer, instances, batch_size=50):
    ndocs = len(instances)

    predictions = []
    for batch_id in tqdm(range(0, ndocs, batch_size), total=ndocs//batch_size, desc='classifying'):
        batch_instances = instances[batch_id:batch_id + batch_size]
        tokenized_dataset = tokenize_function(tokenizer, batch_instances)
        logits = model(**tokenized_dataset).logits
        predicted_class_id = logits.argmax(axis=-1).cpu().numpy()
        predictions.append(predicted_class_id)

    return np.concatenate(predictions)


def get_last_checkpoint(path):
    """
    Searches for the last checkpoint and returns the complete path to it
    :param path: a folder containing all the checkpoints generated during the finetuning
    :return: a path to the last checkpoint (the one with the highest step)
    """
    max_step = np.max([int(checkpoint_folder.split('-')[-1]) for checkpoint_folder in glob(join(path, 'checkpoint-*'))])
    return join(path, f'checkpoint-{max_step}')


def choose_best_epoch(json_path, score, higher_is_better):
    with open(json_path, 'rt') as fin:
        js = json.load(fin)

    tr_states = js['log_history']
    tr_states = [state for state in tr_states if 'eval_macro-f1' in state]

    epoch, step, eval_Mf1 = zip(*[(state['epoch'], state['step'], state[score]) for state in tr_states])

    if higher_is_better:
        best_epoch_pos = np.argmax(eval_Mf1)
    else:
        best_epoch_pos = np.argmin(eval_Mf1)

    return int(epoch[best_epoch_pos]), int(step[best_epoch_pos]), eval_Mf1[best_epoch_pos]


def del_checkpoints(path, keep, rename):
    for checkpoint in glob(join(path, 'checkpoint-*')):
        if checkpoint != keep:
            shutil.rmtree(checkpoint)
        else:
            shutil.move(checkpoint, join(path, rename))


def clean_checkpoints(finetuned_path, score='eval_macro-f1', higher_is_better=True, verbose=True):
    def vprint(msg):
        if verbose:
            print(msg)

    vprint(f'Cleaning folder {finetuned_path}')

    checkpoint_path = get_last_checkpoint(finetuned_path)
    vprint(f'> last checkpoint found: {checkpoint_path}')

    json_path = join(checkpoint_path, 'trainer_state.json')
    epoch, step, bestscore = choose_best_epoch(json_path, score=score, higher_is_better=higher_is_better)
    vprint(f'> best epoch (in terms of {score} --{"higher is better" if higher_is_better else "lower is better"}): {epoch}, step={step}, with {score}={bestscore}')

    best_checkpoint = join(finetuned_path, f'checkpoint-{step}')

    vprint(f'> retaining {best_checkpoint} (renamed as best_checkpoint) and cleaning the rest')
    del_checkpoints(finetuned_path, keep=best_checkpoint, rename='best_checkpoint')
