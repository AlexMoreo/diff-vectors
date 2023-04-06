import numpy as np
import pandas as pd
from glob import glob

logpath = '../log'

filter_suffix=['tmp']
dfs = []
for result in glob('../log/results_arxiv*.csv'):
    read_it = True
    for suff in filter_suffix:
        if result.endswith(f'{suff}.csv'):
            read_it=False
    if not read_it: continue
    df = pd.read_csv(result, sep='\t')
    dfs.append(df)
df = pd.concat(dfs)

df.method = df.method.apply(lambda x: x.replace('PairLRknn', 'Lazy AA'))
df.method = df.method.apply(lambda x: x.replace('PairLRlinear', 'Stacked AA'))
df.method = df.method.apply(lambda x: x.replace('LRbin', 'STD-Bin'))
df.method = df.method.apply(lambda x: x.replace('LR', 'STD-AA'))

from scipy.stats import ttest_ind_from_stats


def ttest(mean1,std1, count1, mean2, std2, count2):
    _,pvalues = ttest_ind_from_stats(mean1, std1, count1, mean2, std2, count2)
    def symbol(pvalue):
        if pvalue==1:
            return '='
        if pvalue > 0.05:
            return '**'
        elif pvalue > 0.005:
            return '*'
        return ''
    return np.array([symbol(pval) for pval in pvalues])


def generate_latex_table(df, path, metric='f1'):
    pv = df.pivot_table(index='method', values=metric, aggfunc=[np.mean, np.std, np.size]).reset_index()

    pos_max = pv['mean'][metric].idxmax()

    max_mean = pv.iloc[pos_max]['mean'][metric]
    max_std  = pv.iloc[pos_max]['std'][metric]
    max_size = pv.iloc[pos_max]['size'][metric]

    pv['ttest']=ttest(pv['mean'][metric], pv['std'][metric], pv['size'][metric], max_mean, max_std, max_size)
    pv = pv.drop(columns='size', level=0)
    open(path, 'wt').write(pv.to_latex(index=False, bold_rows=True, float_format="{:0.3f}".format))


metric='f1_attr_macro'
generate_latex_table(df, f'../latex/attr_arxiv_{metric}-r1.tex', metric=metric)

metric='f1_attr_micro'
generate_latex_table(df, f'../latex/attr_arxiv_{metric}-r1.tex', metric=metric)

# metric='f1_verif'
# generate_latex_table(df, f'../latex/attr_arxiv_{metric}-r1.tex', metric=metric)

# metric='acc_verif'
# generate_latex_table(df, f'../latex/attr_arxiv_{metric}-r1.tex', metric=metric)


