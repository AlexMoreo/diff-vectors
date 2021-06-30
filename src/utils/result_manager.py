import os
import pandas as pd


class AbsResult:

    def __init__(self, path, *columns):
        # open log file
        if os.path.exists(path):
            self.csv = open(path, 'at')
            print('CSV file exists, opening')
        else:
            self.csv = open(path, 'wt')
            print('CSV file does not exist, creating')
            columns = '\t'.join(columns)
            tee(self.csv, f'{columns}')

    def close(self):
        self.csv.close()


class AttributionResult(AbsResult):
    def __init__(self, path):
        super(AttributionResult, self).__init__(
            path, 'dataset','seed', 'n_authors', 'docs_by_author', 'method', 'trainsize', 'time_vec', 'time_tr',
            'time_te', 'acc_attr', 'f1_attr_macro', 'f1_attr_micro', 'acc_verif', 'f1_verif')

    def append(self, dataset, seed, n_authors, docs_by_author, method, num_el,
               vec_time, train_time, test_time, acc_attr, f1_attr_macro,
               f1_attr_micro, acc_verif, f1_verif):
        config = num2str([seed, n_authors, docs_by_author, method, num_el])
        timings = float2str([vec_time, train_time, test_time])
        results = float2str([acc_attr, f1_attr_macro, f1_attr_micro, acc_verif, f1_verif])
        output = [dataset] + config + timings + results
        output = '\t'.join(output)
        tee(self.csv, output)


class SAVResult(AbsResult):
    def __init__(self, path):
        super(SAVResult, self).__init__(
            path, 'dataset', 'seed', 'mode', 'n_authors', 'docs_by_author', 'n_authors_te', 'docs_by_author_te',
            'method', 'acc', 'p', 'r', 'f1', 'notes')

    def append(self, dataset, seed, mode, n_authors, docs_by_author, n_authors_te, docs_by_author_te, method, acc, p, r, f1, notes=""):
        config = num2str([seed, mode, n_authors, docs_by_author, n_authors_te, docs_by_author_te, method])
        results = float2str([acc, p, r, f1])
        output = [dataset] + config + results + [notes]
        output = '\t'.join(output)
        tee(self.csv, output)


def check_if_already_performed(path, dataset, seed, n_authors, docs_by_author, method):
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path, sep='\t')
    except pd.errors.EmptyDataError:
        return False
    return not df[df.dataset==dataset][df.seed==seed][df.n_authors==n_authors][df.docs_by_author==docs_by_author][df.method==method].empty


def count_results(path, dataset, seed, n_authors):
    if not os.path.exists(path):
        return 0
    df = pd.read_csv(path, sep='\t')
    return df[df.dataset==dataset][df.seed==seed][df.n_authors==n_authors].shape[0]


def num2str(x):
    return [f'{xi}' for xi in x]


def float2str(x):
    return [f'{xi:.4f}' for xi in x]


def time2str(x):
    return [f'{xi:.1f}' for xi in x]


def tee (csv, msg):
    csv.write(f'{msg}\n')
    csv.flush()
    print(msg)


# def AttributionResult2SAVResult(path_in, path_out):
#     att = open(path_in, 'rt')
#     sav = SAVResult(path_out)
#     for line in att.readlines()[1:]:
#         parts = line.strip().split('\t')
#         print(parts)
#         dataset, seed, n_authors, docs_by_author, method, _, _, _, _, _, _, _, acc, f1 = parts
#         acc = float(acc)
#         f1 = float(f1)
#         mode = 'OPEN' if 'open' in method else 'CLOSE'
#         sav.append(dataset, seed, mode, n_authors, docs_by_author, 10, 25, method, acc, f1)
#
# AttributionResult2SAVResult('../../log-sav/results_victorian.csv', '../../log-sav/results_victorian.csv')