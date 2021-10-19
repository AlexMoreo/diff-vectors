from abc import ABC, abstractmethod
import random
import numpy as np
from collections import Counter


class LabelledCorpus:

    def __init__(self, documents, labels):
        documents = np.asarray(documents)
        labels = np.asarray(labels)
        self.data = documents
        self.target = labels

    def __len__(self):
        return len(self.data)

    @classmethod
    def filter(cls, labelled_corpus, to_drop):
        sel_data, sel_target = [], []
        for i in range(len(labelled_corpus)):
            if labelled_corpus.target[i] not in to_drop:
                sel_data.append(labelled_corpus.data[i])
                sel_target.append(labelled_corpus.target[i])
        return LabelledCorpus(sel_data, sel_target)


class AuthorshipDataset(ABC):

    def __init__(self, data_path, n_authors=-1, docs_by_author=-1, n_open_set_authors = 0, random_state=42):
        self.data_path = data_path

        random.seed(random_state)
        np.random.seed(random_state)

        self._check_n_authors(n_authors, n_open_set_authors)

        self.train, self.test, self.target_names = self._fetch_and_split()

        self._assure_docs_by_author(docs_by_author)

        self._reduce_authors_documents(n_authors, docs_by_author, n_open_set_authors)

        self._remove_label_gaps()

        super().__init__()


    @abstractmethod
    def _fetch_and_split(self):
        pass


    @abstractmethod
    def _check_n_authors(self, n_authors, n_open_set_authors):
        pass


    def _reduce_authors_documents(self, n_authors, n_docs_by_author, n_open_set_authors):

        if n_authors != -1 or n_docs_by_author != -1:
            #training data only (test contains all examples by author)
            if n_docs_by_author != -1:
                docs_by_author = self.group_by(self.train.data, self.train.target)
                train_labels, train_data = [], []
                for author, documents in docs_by_author.items():
                    if n_docs_by_author > len(documents):
                        continue
                    selected_docs = random.sample(documents, n_docs_by_author)
                    train_labels.extend([author] * n_docs_by_author)
                    train_data.extend(selected_docs)

                self.train = LabelledCorpus(train_data, train_labels)

            if n_authors == -1:
                selected_authors = self.target_names
            else:
                selected_authors = random.sample(self.target_names, n_authors+n_open_set_authors)
                self.test = self.extract_documents_from_authors(self.test, selected_authors)
                self.train = self.extract_documents_from_authors(self.train, selected_authors)
        else:
            selected_authors = np.unique(self.train.target)

        if n_open_set_authors > 0:
            self.train, self.test, self.test_out = self.disjoint_train_test_authors(
                self.train, self.test, n_open_set_authors, selected_authors
            )
        else:
            self.test_out = None


    # reindex labels so that the unique labels are equal to range(#num_different_authors)
    # and unique training labels are range(#num_different_training_authors)
    def _remove_label_gaps(self):

        # reindex the training labels first, so that they contain no gaps
        unique_labels = np.unique(self.train.target)
        recode={old:new for old,new in zip(unique_labels,range(len(unique_labels)))}
        self.train.target=np.array([recode[l] for l in self.train.target])
        self.test.target = np.array([recode[l] for l in self.test.target])

        #test_out_labels (if requested) contains additional authors
        if self.test_out is not None:
            for l in np.unique(self.test_out.target):
                if l not in recode:
                    recode[l] = len(recode)
            self.test_out.target = np.array([recode[l] for l in self.test_out.target])

    def group_by(self, docs, authors):
        return {i: docs[authors == i].tolist() for i in np.unique(authors)}

    def extract_documents_from_authors(self, labelled_docs, authors):
        X, y = labelled_docs.data, labelled_docs.target
        if not isinstance(X, np.ndarray): X = np.asarray(X)
        if not isinstance(y, np.ndarray): y = np.asarray(y)
        idx = np.logical_or.reduce([y == i for i in authors])
        return LabelledCorpus(X[idx], y[idx])

    def disjoint_train_test_authors(self, train, test, n_open_test_authors, selected_authors):
        train_authors, test_authors = selected_authors[n_open_test_authors:], selected_authors[:n_open_test_authors]

        train = self.extract_documents_from_authors(train, train_authors)
        test_in  = self.extract_documents_from_authors(test, train_authors)
        test_out = self.extract_documents_from_authors(test, test_authors)

        return train, test_in, test_out

    def _assure_docs_by_author(self, docs_by_author):
        if docs_by_author == -1:
            return

        author_doc_count = Counter(self.train.target)
        to_remove = frozenset([id for id,count in author_doc_count.most_common() if count<docs_by_author])
        assert len(to_remove) < len(author_doc_count), 'impossible selection'
        if len(to_remove)>0:
            self.train = LabelledCorpus.filter(self.train, to_remove)
            self.test  = LabelledCorpus.filter(self.test,  to_remove)
            self.target_names = sorted(set(self.target_names) - to_remove)


