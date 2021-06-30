import numpy as np
from sklearn.model_selection import train_test_split
import random

from data.AuthorshipDataset import AuthorshipDataset, LabelledCorpus


class Imdb62(AuthorshipDataset):

    TEST_SIZE = 0.30
    NUM_AUTHORS = 62
    NUM_DOCS_BY_AUTHOR = int(1000-(1000*TEST_SIZE))

    def __init__(self, data_path='../data/imdb62/imdb62.txt', n_authors=-1, docs_by_author=-1, n_open_set_authors = 0, random_state=42):
        super().__init__(data_path, n_authors, docs_by_author, n_open_set_authors, random_state)


    def _fetch_and_split(self):
        file = open(self.data_path,'rt', encoding= "utf-8").readlines()
        splits = [line.split('\t') for line in file]
        reviews = np.asarray([split[4]+' '+split[5] for split in splits])

        authors=[]
        authors_ids = dict()
        for s in splits:
            author_key = s[1]
            if author_key not in authors_ids:
                authors_ids[author_key]=len(authors_ids)
            author_id = authors_ids[author_key]
            authors.append(author_id)
        authors = np.array(authors)

        authors_names = sorted(np.unique(authors))

        train_data, test_data, train_labels, test_labels = \
            train_test_split(reviews, authors, test_size=Imdb62.TEST_SIZE, stratify=authors)

        return LabelledCorpus(train_data, train_labels), LabelledCorpus(test_data, test_labels), authors_names


    def _check_n_authors(self, n_authors, n_open_set_authors):
        if n_authors==-1: return
        elif n_authors+n_open_set_authors > Imdb62.NUM_AUTHORS:
            raise ValueError(f'Too many authors requested. Max is {Imdb62.NUM_AUTHORS}')

