import numpy as np
import xml.etree.ElementTree
from data.AuthorshipDataset import LabelledCorpus, AuthorshipDataset

"""
subset refers either to small or large collection
corpus refers to the dataset, either "devel" (the train-test split made available before the PAN2011 competition) or
"competititon" (the train-test split on which methods were evaluated in PAN2011)
"""
class Pan2011(AuthorshipDataset):

    FOLDER={'devel':'pan11-author-identification-training-corpus-2011-04-08',
            'competition':'pan11-author-identification-test-corpus-2011-05-23'}

    FILE={'small':('SmallTrain.xml','SmallTest.xml','GroundTruthSmallTest.xml'),
          'large':('LargeTrain.xml','LargeTest.xml','GroundTruthLargeTest.xml')}


    def __init__(self, data_path='../data', n_authors=-1, docs_by_author=-1, n_open_set_authors=0, random_state=42):
        super().__init__(data_path, n_authors, docs_by_author, n_open_set_authors, random_state)


    def _fetch_and_split(self):
        path=self.data_path
        subset='large'
        min_length=15
        train, test, keys = Pan2011.FILE[subset]

        train_texts = fetch_xml(f'{path}/{Pan2011.FOLDER["devel"]}/{train}', valid=False, min_length=min_length)
        test_texts = fetch_xml(f'{path}/{Pan2011.FOLDER["competition"]}/{test}', valid=True, min_length=min_length)
        test_labels = fetch_groundtruth(f'{path}/{Pan2011.FOLDER["competition"]}/{keys}')
        test_texts = attatch_labels(test_texts, test_labels)

        train_data, train_labels = list(zip(*list(train_texts.values())))
        test_data, test_labels = list(zip(*list(test_texts.values())))

        author_index = {author:i for i,author in enumerate(np.unique(train_labels))}
        train_labels = [author_index[author] for author in train_labels]
        test_labels = [author_index[author] for author in test_labels]
        target_names = sorted(np.unique(train_labels + test_labels))

        return LabelledCorpus(train_data, train_labels), LabelledCorpus(test_data, test_labels), target_names


    def _check_n_authors(self, n_authors, n_open_set_authors):
        pass



def fetch_xml(path, valid=False, min_length=10):
    # corpus = xml.etree.ElementTree.parse(path).getroot() # this doesn't work due to inconsistencies found within body elements
    lines = list(map(str.strip, open(path, 'rt').readlines()))
    texts = {}
    rootname = 'training' if valid==False else 'testing'
    assert lines[0]==f'<{rootname}>', f'wrong root {lines[0]}'

    i=1
    while lines[i]!=f'</{rootname}>':
        if lines[i].startswith('<text file='):
            l=lines[i]
            textid = l[l.index('"')+1:l.rindex('"')]
            i+=1
            author_id = None
            while lines[i] != '</text>':
                l=lines[i]
                if l.startswith('<author id='):
                    author_id = l[l.index('"')+1:l.rindex('"')]
                elif l.startswith('<body>'):
                    i+=1
                    body=[]
                    while lines[i]!='</body>':
                        body.append(lines[i])
                        i+=1
                    body = '\n'.join(body)
                i+=1

            if min_length == -1 or len(body.split()) >= min_length:
                texts[textid] = (body, author_id)

        i+=1

    print(f'read {len(texts)} documents')
    return texts


def fetch_groundtruth(path):
    authors = {}
    results = xml.etree.ElementTree.parse(path).getroot()
    for result in results.findall('text'):
        file = result.attrib['file']
        file = file[file.find('"')+1:]
        author = result.find('author').attrib['id']
        authors[file] = author

    return authors


def attatch_labels(texts, authors):
    for textid in texts.keys():
        text, _ = texts[textid]
        texts[textid] = (text, authors[textid])
    return texts



