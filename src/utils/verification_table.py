import pandas as pd
import numpy as np
from scipy.stats import ttest_ind_from_stats
from glob import glob

import sys


logattr = '../logAVfor10authors'
n_authors = 10

dfs = []
for result in glob(f'{logattr}/results_AV_*.csv'):
    # if 'imdb62' not in result:
        df = pd.read_csv(result, sep='\t')
        dataset = df.dataset.unique()[0]
        print(result, dataset)
        df = df[df.n_authors == n_authors]
        dfs.append(df)
df = pd.concat(dfs)

DATASETS = ['IMDb62', 'PAN2011', 'Victorian', 'arXiv']
METHODS = ['StdVectors', 'DiffVectors-knn', 'DiffVectors-lin']
df = df.replace('LRbin', 'StdVectors')
df = df.replace('PairLRknnbin', 'DiffVectors-knn')
df = df.replace('PairLRlinearbin', 'DiffVectors-lin')
df = df.replace('arxiv', 'arXiv')
df = df.replace('pan2011', 'PAN2011')
df = df.replace('imdb62', 'IMDb62')
df = df.replace('victorian', 'Victorian')


def get_func_by_dataset(df, func, func_name):
    print(df)
    pv_aggr = df.pivot_table(
        index='method',
        columns='dataset',
        values='f1_verif',
        aggfunc=func
    )
    print(pv_aggr)
    plain = pv_aggr.reset_index()
    print(plain)
    # methods = plain.method.values
    # values = plain.arxiv.values
    datasets = plain.columns.values[1:]
    data = {dataset:{} for dataset in datasets}
    for dataset in datasets:
        for method, value in zip(plain.method.values, plain[dataset].values):
            data[dataset][method] = value
    return data


mean = get_func_by_dataset(df, np.mean, 'mean')
std = get_func_by_dataset(df, np.std, 'std')
nd = get_func_by_dataset(df, np.size, 'nd')

print(mean)
print(std)
print(nd)
print(DATASETS)

best = {}
for dataset in DATASETS:
    method_names = list(mean[dataset].keys())
    method_results = list(mean[dataset].values())
    best_pos = np.argmax(method_results)
    best_method = method_names[best_pos]
    print(best_pos, best_method)
    best[dataset] = best_method

with open('../latex/AV.tex', 'wt') as foo:
    foo.write('\\begin{tabular}{|l'+ ('|rrr'*len(DATASETS)) +'|}')
    foo.write('\hline\n')
    # foo.write(' & ' + ' & '.join("\\texttt{"+d+"}" for d in DATASETS))
    foo.write(' & ' + ' & '.join("\multicolumn{3}{c|}{\\texttt{" + d + "}}" for d in DATASETS))
    foo.write(' \\\\\n')
    foo.write('\hline\n')
    foo.write(' & mean & std & ttest'*len(DATASETS))
    foo.write(' \\\\\n')
    foo.write('\hline\n')
    for method in METHODS:
        foo.write(f'{method} ')
        for dataset in DATASETS:
            value = mean[dataset][method]
            value_str = f'{value:.3f}'
            std_str = f'{std[dataset][method]:.3f}'
            is_best = best[dataset] == method
            ttest = ''
            if is_best:
                value_str = "\\textbf{" + value_str + '}'
            else:
                best_mean = mean[dataset][best[dataset]]
                best_std = std[dataset][best[dataset]]
                best_nd = nd[dataset][best[dataset]]
                this_mean = mean[dataset][method]
                this_std = std[dataset][method]
                this_nd = nd[dataset][method]
                _, p_val = ttest_ind_from_stats(best_mean, best_std, best_nd, this_mean, this_std, this_nd)
                print(p_val)
                if p_val > 0.05:
                    ttest = '**'
                elif p_val > 0.001:
                    ttest = '*'
            foo.write(f' & {value_str} & {std_str} & {ttest}')

        foo.write(' \\\\\n')
    foo.write('\hline\n')
    foo.write('\end{tabular}')

sys.exit(0)



