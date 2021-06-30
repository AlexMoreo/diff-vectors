import jsonlines
import numpy as np
from sklearn.model_selection import train_test_split
from data.AuthorshipDataset import LabelledCorpus, AuthorshipDataset
from collections import Counter


class Pan2020 (AuthorshipDataset):

    def __init__(self, data_path='../data/pan20-authorship-verification-training-small/', n_authors=-1,
                 docs_by_author=-1, n_open_set_authors=0, random_state=42):
        super().__init__(data_path, n_authors, docs_by_author, n_open_set_authors, random_state)

    def _fetch_and_split(self):

        data = []
        with jsonlines.open(self.data_path + 'pan20-authorship-verification-training-small.jsonl') as f:
            for line in f.iter():
                for text in line['pair']:
                    data.append(text)

        ids = []
        with jsonlines.open(self.data_path + 'pan20-authorship-verification-training-small-truth.jsonl') as f:
            for line in f.iter():
                for author in line['authors']:
                    ids.append(int(author))

        # retain only texts with a number of tokens between [10,10000)
        data, ids = zip(*[(t, id) for t, id in zip(data, ids) if 10 <= len(t.split()) < 10000])

        # retain only texts of authors with at least 2 documents
        counter = Counter(ids)
        data, ids = zip(*[(t,id) for t,id in zip(data, ids) if counter[id]>1])

        author_ids = sorted(np.unique(ids))

        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, ids, test_size=0.30, stratify=ids)

        return LabelledCorpus(train_data, train_labels), LabelledCorpus(test_data, test_labels), author_ids

    def _check_n_authors(self, n_authors, n_open_set_authors):
        pass