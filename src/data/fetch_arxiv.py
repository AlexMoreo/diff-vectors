import urllib.request as libreq
#import feedparser
import time
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from data.AuthorshipDataset import AuthorshipDataset, LabelledCorpus


class Arxiv(AuthorshipDataset):

    TEST_SIZE = 0.3

    def __init__(self, data_path='../data/arXiv/request_merged.pickle', n_authors=-1, docs_by_author=-1, n_open_set_authors=0, random_state=42):
        super().__init__(data_path, n_authors, docs_by_author, n_open_set_authors, random_state)


    def _fetch_and_split(self):
        assert os.path.exists(self.data_path), 'Pickled object not found. ' \
                                          'First create it by running "python ./data/fetch_arxiv.py"'

        request = pickle.load(open(self.data_path, 'rb'))

        data = ['\n'.join([title, abstract]) for (id,title,abstract,author) in request]
        labels = [author for (id,title,abstract,author) in request]

        target_names = sorted(np.unique(labels))

        train_data, test_data, train_labels, test_labels = \
            train_test_split(data, labels, test_size=Arxiv.TEST_SIZE, stratify=labels)

        return LabelledCorpus(train_data, train_labels), LabelledCorpus(test_data, test_labels), target_names


    def _check_n_authors(self, n_authors, n_open_set_authors):
        pass


"""
Given a list of query terms (e.g., 'machine learning'), this function composes the query using the arXiv API and 
harvests the responses. Only responses corresponding to single-authored papers are taken, and from them, the id,
title, abstract, and author name is retained.
This function is quite slow. The main reason being that the arXiv's API recommends to add a "sleep" between subsequent
requests in order not to overload the server.
The API gives access to no more than 30,000 items per query. It is thencefore better to make separate queries (e.g.,
one for 'machine learning' and one for 'deep learning') and merge them afterwards than make a single query (e.g., 
'machine learning and deep learning'). In case of merging them, remember to filter out duplicates. 
For some reason, sometimes the response is empty (this has been noticed in the discussion forum). We simply retry those
a maximum of 10 times.
"""
def crawl_single_authored_papers(abstract_terms_query):
    print('fetching arXiv single-authored title+abstracts')

    search_query = '+OR+'.join([f'abs:{term}' for term in abstract_terms_query])

    # This is a hack to expose both of these namespaces in feedparser v4.1
    feedparser._FeedParserMixin.namespaces['http://a9.com/-/spec/opensearch/1.1/'] = 'opensearch'
    feedparser._FeedParserMixin.namespaces['http://arxiv.org/schemas/atom'] = 'arxiv'

    start = 0  # retreive the first 5 results
    max_results = 500
    total = 30000
    found = 0

    request = []

    redo=0
    while total is None or start < total:
        query = 'search_query=%s&start=%i&max_results=%i' % (search_query, start, max_results)

        # perform a GET request using the base_url and query
        with libreq.urlopen(base_url + query) as url:
          response = url.read()

        # parse the response using feedparser
        feed = feedparser.parse(response)

        # print out feed information
        if total is None:
            print('Feed title: %s' % feed.feed.title)
            print('Feed last updated: %s' % feed.feed.updated)
            print('totalResults for this query: %s' % feed.feed.opensearch_totalresults)
            print('itemsPerPage for this query: %s' % feed.feed.opensearch_itemsperpage)
            print('startIndex for this query: %s' % feed.feed.opensearch_startindex)

        total = min(total, int(feed.feed.opensearch_totalresults))

        n_authors_list = []
        n_entries = len(feed.entries)
        for entry in feed.entries:
            try:
                n_authors = len(entry.authors)
                if n_authors==1:
                    found += 1
                    arxivid = entry.id.split('/abs/')[-1]
                    title = entry.title
                    author_name = entry.authors[0].name
                    abstract = entry.summary
                    request.append((arxivid, title, abstract, author_name))
                n_authors_list.append(n_authors)
            except AttributeError:
                pass

        if n_entries==0 and redo < 10:
            redo+=1
        else:
            start += max_results
            redo=0
        n_author_ave = np.mean(n_authors_list) if n_authors_list else 0

        print(f'parsed {start}/{total} ({100*start/total:.1f}% complete): found {found} '
              f'[entries={n_entries} average in batch={n_author_ave}][redo={redo}]')

        # playing nice :)
        time.sleep(5)

    return request


"""
This script will harvest responses for different queries independently, and then will merge the responses toghether
removing duplicates.
This takes quite some time.
"""
if __name__ == '__main__':

    data_path = '../data/arXiv'

    base_url = 'http://export.arxiv.org/api/query?';

    abstract_terms_query = ['deep+learning', 'machine+learning', 'information+retrieval', 'computer+science',
                            'data+mining', 'support+vector', 'logistic+regression', 'artificial+intelligence',
                            'supervised+learning']

    # for abstract_term_query in abstract_terms_query:
    #     print(f'crawling for {abstract_term_query}')
    #     request = crawl_single_authored_papers([abstract_term_query])
    #
    #     print('pickling...')
    #     pickle.dump(request, open(f'{data_path}/request_{abstract_term_query}.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    print('arXiv crawl done. Merging results.')
    merged = []
    id_register = set()

    for abstract_term_query in abstract_terms_query:
        request = pickle.load(open(f'{data_path}/request_{abstract_term_query}.pickle', 'rb'))
        no_duplicates = [r for r in request if r[0] not in id_register]
        reqsize = len(request)
        nodupsize = len(no_duplicates)
        print(f'load {reqsize} ({nodupsize} new) entries for query={abstract_term_query}')

        merged.extend([r for r in request if r[0] not in id_register])
        id_register.update([id for id, _, _, _ in request])

    print(f'total entries without duplicates is {len(merged)}')
    min_docs = 10
    author_count = Counter([author for _,_,_,author in merged if 'Collaboration' not in author]).most_common()
    author_10docs = frozenset([author for author,count in author_count if count >= min_docs])
    merged = [paper for paper in merged if paper[-1] in author_10docs]

    print(f'authors with at least {min_docs} papers {len(author_10docs)}')

    print(f'pickling merged requests (final size = {len(merged)})')
    pickle.dump(merged, open(f'{data_path}/request_merged.pickle', 'wb'), pickle.HIGHEST_PROTOCOL)

    total=0
    for i,(author,count) in enumerate(author_count):
        if count >= min_docs:
            print(f'{i} {author}\t{count} papers')
            total+=count
    print(f'Total articles: {total}')




