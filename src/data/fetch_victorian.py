import numpy as np
import csv
from sklearn.model_selection import train_test_split
from data.AuthorshipDataset import AuthorshipDataset, LabelledCorpus


class Victorian(AuthorshipDataset):

    TEST_SIZE = 0.30

    def __init__(self, data_path='../data/victoria', n_authors=-1, docs_by_author=-1, n_open_set_authors=0, random_state=42):
        super().__init__(data_path, n_authors, docs_by_author, n_open_set_authors, random_state)

    def _fetch_and_split(self):

        data, labels = [], []

        with open (f'{self.data_path}/Gungor_2018_VictorianAuthorAttribution_data-train.csv','r',encoding="latin-1") as file:
            csv_reader = csv.reader(file, delimiter = ',')
            next(csv_reader)
            for row in csv_reader:
                # if row[0]!='text':
                data.append(row[0])
                labels.append(int(row[1]))

        target_names = sorted(np.unique(labels))

        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=Victorian.TEST_SIZE, stratify=labels)

        return LabelledCorpus(train_data, train_labels), LabelledCorpus(test_data, test_labels), target_names


    def _check_n_authors(self, n_authors, n_open_set_authors):
        pass