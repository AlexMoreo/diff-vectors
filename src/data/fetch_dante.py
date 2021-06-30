import os
import numpy as np


class Dante_mix:

    def __init__(self, path_dante_mix='../data/dante_mix'):
        self.data_path = path_dante_mix

        self.data = []
        self.labels = []

        authors = []
        for file in os.listdir(self.data_path):
            file_clean = file.replace('.txt', '')
            author, textname = file_clean.split('_')[0], file_clean.split('_')[1]
            if author not in authors:
                authors[author] = []
            authors[author].append(textname)
