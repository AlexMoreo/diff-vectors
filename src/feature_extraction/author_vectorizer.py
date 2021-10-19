import nltk
from nltk.corpus import stopwords
import collections
import numpy as np
from scipy.sparse import issparse, csr_matrix
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import string
import itertools
from joblib import Parallel, delayed
import multiprocessing
from utils.common import StandardizeTransformer, feature_selection


# ------------------------------------------------------------------------
# CLEAN TEXTS
# ------------------------------------------------------------------------
def clean_texts(texts):
    return [_remove_citations(text) for text in texts]


#remove unwanted pattern from text
def _remove_pattern(text, start_symbol, end_symbol, counter):
    assert counter[start_symbol] == counter[end_symbol], f'wrong number of {start_symbol}{end_symbol} found'
    search = True
    while search:
        start = text.find(start_symbol)
        if start > -1:
            end = text[start + 1:].find(end_symbol)
            text = text [:start] + text [start + 1 + end + 1:]
        else:
            search = False
    return text


# remove citations in format: *latino* | {volgare}
def _remove_citations(text):
    counter = collections.Counter(text)
    text = _remove_pattern(text, start_symbol='*', end_symbol='*', counter=counter)
    text = _remove_pattern(text, start_symbol='{', end_symbol='}', counter=counter)
    return text


# ------------------------------------------------------------------------
#THIS CLASS CREATES THE TABLE CONTAINING THE FEATURE VALUES
# ------------------------------------------------------------------------
function_words_dict = {
    'latin' : ['et',  'in',  'de',  'ad',  'non',  'ut', 'cum', 'per', 'a', 'sed', 'que', 'quia', 'ex', 'sic',
                'si', 'etiam', 'idest', 'nam', 'unde', 'ab', 'uel', 'sicut', 'ita', 'enim', 'scilicet', 'nec',
                'pro', 'autem', 'ibi',  'dum', 'uero', 'tamen', 'inter', 'ideo', 'propter', 'contra', 'sub',
                'quomodo', 'ubi', 'super', 'iam', 'tam', 'hec', 'post', 'quasi', 'ergo', 'inde', 'e', 'tunc',
                'atque', 'ac', 'sine', 'nisi', 'nunc', 'quando', 'ne', 'usque', 'siue', 'aut', 'igitur', 'circa',
                'quidem', 'supra', 'ante', 'adhuc', 'seu' , 'apud', 'olim', 'statim', 'satis', 'ob', 'quoniam',
                'postea', 'nunquam'
                ],
}


class FeatureExtractor:
    def __init__(self, lang, cleaning=False, standardize=True, max_sparse_features=50000, use_raw_frequencies=False,
                 function_words=True, word_lengths=True, sentence_lengths=True, post_ngrams=True, punctuation=True, word_ngrams=True, char_ngrams=True):
        """
        A stylometric-based feature extractor for authorship analysis.
        :param lang: the language of the input documents (currently, only Latin and English is supported)
        :param cleaning: whether or not to clean editor's comments and quotations (not used)
        :param standardize: whether or not to standardize (i.e., zero-mean and unit-varianze) the dense feature subsets
        :param max_sparse_features: if the size of the union of the sparse feature subsets exceeds this value, a
            feature selection process (filter-stile with Chi^2 TSR) will be invoked to reduce it up to this value
        :param use_raw_frequencies: whether or not to use tfidf weighting or raw frequencies
        """
        self.lang = lang
        self.cleaning = cleaning
        self.standardize=standardize
        self.max_num_feats = max_sparse_features
        self.use_raw_frequencies=use_raw_frequencies
        self.function_words = function_words
        self.word_lengths = word_lengths
        self.sentence_lengths = sentence_lengths
        self.post_ngrams = post_ngrams
        self.punctuation = punctuation
        self.word_ngrams = word_ngrams
        self.char_ngrams = char_ngrams

    def tokenize(self, text):
        unmod_tokens = nltk.word_tokenize(text)
        return ([token.lower() for token in unmod_tokens if any(char.isalpha() for char in token)])

    # ---funcction words
    def _features_function_words_freq(self, tokens):
        feats = []
        function_words = function_words_dict.get(self.lang, stopwords.words(self.lang))
        for doc in tokens:
            freqs = nltk.FreqDist(doc)
            nwords = max(1,len(doc))
            funct_words_freq = [freqs[function_word] / nwords for function_word in function_words]
            feats.append(funct_words_freq)
        return np.asarray(feats)

    # ---word lengths
    def _features_word_lengths(self, tokens, upto=23):
        feats = []
        for doc in tokens:
            nwords = max(1,len(doc))
            tokens_len = [len(token) for token in doc]
            tokens_count = []
            for i in range(1, upto):
                tokens_count.append((sum(j >= i for j in tokens_len)) / nwords)
            feats.append(tokens_count)
        return np.asarray(feats)

    # ---sentence lengths
    def _features_sentence_lengths(self, documents, downto=3, upto=70):
        feats = []
        for doc in documents:
            sentences = [t.strip() for t in nltk.tokenize.sent_tokenize(doc) if t.strip()]
            nsent = max(1,len(sentences))
            sent_len = [len(self.tokenize(sentence)) for sentence in sentences]
            sent_count = []
            for i in range(downto, upto):
                sent_count.append(sum(j >= i for j in sent_len) / nsent)
            feats.append(sent_count)
        return np.asarray(feats)

    # ---postags ngrams
    def pos_tagger(self, documents):
        lang_model = {
            'english' : 'en_core_web_sm',
            'spanish' : 'es_core_news_sm'
        }
        n_jobs = multiprocessing.cpu_count()
        n_docs = len(documents)
        batch = int(n_docs / n_jobs)

        tags_docs = Parallel(n_jobs=-1)(
            delayed(pos_tagger_task)(
                documents[job*batch : (job+1)*batch + (n_docs % n_jobs if job == n_jobs - 1 else 0)], job, lang_model['english'])
            for job in range(n_jobs)
        )

        return list(itertools.chain.from_iterable(tags_docs))

    def fit(self, texts, labels, pos_tags=None):
        if pos_tags is None and self.post_ngrams:
            pos_tags = self.pos_tagger(texts)
        if self.use_raw_frequencies:
            use_idf=False
            norm='l1'
        else:
            use_idf = True
            norm = 'l2'

        if self.post_ngrams:
            self.tfidf_post = TfidfVectorizer(analyzer='word', ngram_range=(3, 4), use_idf=use_idf, norm=norm).fit(pos_tags)
        if self.punctuation:
            self.tfidf_punctuation = TfidfVectorizer(analyzer='char', vocabulary=string.punctuation, use_idf=use_idf, norm=norm, min_df=3).fit(texts)
        if self.word_ngrams:
            self.tfidf_words = TfidfVectorizer(analyzer='word', use_idf=use_idf, norm=norm, min_df=3).fit(texts)
        if self.char_ngrams:
            self.tfidf_charngrams = TfidfVectorizer(analyzer='char', ngram_range=(2, 5), use_idf=use_idf, norm=norm).fit(texts)
        # self.tfidf_wordngrams = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), use_idf=use_idf, norm=norm).fit(texts)
        if self.standardize:
            self.standardizer_dense = StandardizeTransformer()
        self.feature_selector_sparse = SelectKBest(chi2, k=self.max_num_feats)

        return self

    def transform(self, texts, labels=None, pos_tags=None):
        densefeatures, sparsefeatures = [], []
        texts = clean_texts(texts) if self.cleaning else texts
        tokens = [self.tokenize(text) for text in texts]
        if pos_tags is None and self.post_ngrams:
            pos_tags = self.pos_tagger(texts)

        # dense features
        if self.function_words:
            densefeatures.append(self._features_function_words_freq(tokens))
        if self.word_lengths:
            densefeatures.append(self._features_word_lengths(tokens))
        if self.sentence_lengths:
            densefeatures.append(self._features_sentence_lengths(texts))
        if len(densefeatures) > 0:
            densefeatures = np.hstack(densefeatures)
            if self.standardize:
                if not self.standardizer_dense.yetfit:
                    print('fitting the standardizer')
                    self.standardizer_dense.fit(densefeatures)
                densefeatures = self.standardizer_dense.transform(densefeatures)

        # sparse features
        if self.post_ngrams:
            sparsefeatures.append(self.tfidf_post.transform(pos_tags))
        if self.punctuation:
            sparsefeatures.append(self.tfidf_punctuation.transform(texts))
        if self.word_ngrams:
            sparsefeatures.append(self.tfidf_words.transform(texts))
        if self.char_ngrams:
            sparsefeatures.append(self.tfidf_charngrams.transform(texts))
        # sparsefeatures.append(self.tfidf_wordngrams.transform(texts))
        if len(sparsefeatures)>0:
            sparsefeatures = scipy.sparse.hstack(sparsefeatures)
            if self.max_num_feats!=-1 and sparsefeatures.shape[1] > self.max_num_feats:
                print('num features orig ', sparsefeatures.shape[1])
                if labels is not None:
                    sparsefeatures = self.feature_selector_sparse.fit_transform(sparsefeatures, labels)
                else:
                    sparsefeatures = self.feature_selector_sparse.transform(sparsefeatures)
        if sparsefeatures != []:
            if len(densefeatures) != []:
                features = scipy.sparse.hstack([sparsefeatures, densefeatures])
            else:
                features = sparsefeatures
        else:
            features = densefeatures

        if issparse(features) and not isinstance(features, csr_matrix):
            features = csr_matrix(features)

        self.sparse_features_range = (0, sparsefeatures.shape[1])
        self.dense_features_range = (sparsefeatures.shape[1], features.shape[1])

        return features

    def fit_transform(self, texts, labels):
        pos_tags = self.pos_tagger(texts) if self.post_ngrams else None
        return self.fit(texts, labels, pos_tags=pos_tags).transform(texts, labels, pos_tags=pos_tags)


def pos_tagger_task(documents, job, lang_model):
    nlp = spacy.load(lang_model, disable=['parser', 'ner'])

    tags_docs = []
    for doc in tqdm(documents, desc=f'POST job {job}'):
        tags = ' '.join(token.pos_ for token in nlp(str(doc)))
        tags_docs.append(tags)

    return tags_docs

