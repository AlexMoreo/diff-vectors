import os
import numpy as np

class Pan_2015:

    def __init__(self, path_pan2015='../data/pan15-authorship-verification-training-dataset-2015-04-19', lang = 'spanish'):

        path_pan2015_lang = f'{path_pan2015}/pan15-authorship-verification-training-dataset-{lang}-2015-04-19'

        self.data = []
        self.labels = []

        for i,dir in enumerate(os.listdir(path_pan2015_lang)):
            if dir in ['contents.json', 'truth.txt']: continue
            for text in os.listdir(f'{path_pan2015_lang}/{dir}'):
                if 'unknown' in text: continue
                self.data.append(open(f'{path_pan2015_lang}/{dir}/{text}','rt', encoding= "utf-8").read())
                self.labels.append(i)

        self.data = np.asarray(self.data)
        self.labels = np.asarray(self.labels)


