import datetime
import json
import eml_parser
import os
from glob import glob
from tqdm import tqdm
import numpy as np


class EnronMail:

    NUM_AUTHORS = 150

    def __init__(self, mail_dir='../data/enron_mail_20150507/maildir/*', n_authors=-1, docs_by_author=-1):
        self.mail_dir = mail_dir
        self._fetch(n_authors, docs_by_author)


    def filter(self, base_str, filter_str):
        if filter_str in base_str:
            idx = base_str.index(filter_str)
            base_str = base_str[:idx]
        return base_str

    def _fetch(self, n_authors, n_docs_by_author, min_tokens=10):

        self.labels = []
        self.data = []

        unique_authors = dict()

        subject_filters = ['fw:','fwd:','re:']
        body_filters = ['-----Original Message-----', '----- Forward',
                        'cc:', 'To:', 'to:', 'From:', 'from:']

        parsed_mails = 0
        author_bar = tqdm(glob(self.mail_dir))
        for author_path in author_bar:
            author_name = author_path[author_path.rindex('/')+1:]
            author_docs = 0
            add_author = False
            author_mails = []
            for email in list(glob(f'{author_path}/sent/*')) + list(glob(f'{author_path}/sent_items/*')):
                author_bar.set_description(f'parsing for {author_path} (docs={len(self.data)}/{parsed_mails} authors={len(unique_authors)}/{150})')
                raw_email = open(email, 'rb').read()
                parsed_mail = eml_parser.eml_parser.decode_email_b(raw_email, include_raw_body=True)
                subject = parsed_mail['header']['subject']
                body = parsed_mail['body'][0]['content']

                for filter in subject_filters:
                    if filter in subject.lower():
                        continue

                for filter in body_filters:
                    body = self.filter(body, filter)

                # body = subject+'\n'+body
                ntokens = len(body.split())
                if ntokens >= min_tokens:
                    if author_name not in unique_authors:
                        unique_authors[author_name]=len(unique_authors)
                    author_idx = unique_authors[author_name]
                    author_mails.append(body)
                    author_docs+=1
                    if n_docs_by_author!=-1 and author_docs>=n_docs_by_author:
                        add_author = True
                        break

                parsed_mails+=1

            if n_docs_by_author==-1 and len(author_mails) >= 5:
                add_author = True

            if add_author:
                self.labels.extend([author_idx]*len(author_mails))
                self.data.extend(author_mails)
            else:
                if author_name in unique_authors:
                    del unique_authors[author_name]

            if len(unique_authors)>=n_authors:
                break

