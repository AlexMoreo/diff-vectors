import numpy as np
import pandas as pd
from glob import glob

from utils.common import ttest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)


def run_ttest_in_table(df):

    df = df.reset_index()

    datasets = df['mean'].columns.values
    for dataset in datasets:

        pos_max = df['mean'][dataset].idxmax()

        max_mean = df.iloc[pos_max]['mean'][dataset]
        max_std  = df.iloc[pos_max]['std'][dataset]
        max_size = df.iloc[pos_max]['size'][dataset]

        df['ttest-'+dataset] = ttest(df['mean'][dataset], df['std'][dataset], df['size'][dataset], max_mean, max_std, max_size)

        df[dataset] = df['ttest-'+dataset] + " " + df['mean'][dataset].round(4).astype(str) + ' $\pm$ ' + df['std'][dataset].round(4).astype(str)

    df = df.drop(columns=['mean','size','std', 'index'], level=0)
    df = df.drop(columns=['ttest-'+dataset for dataset in datasets], level=0)
    df.columns = df.columns.droplevel(1)

    def nice_str(x):
        ttest, mean, pm, std = x.split()
        def pad(x): return x.ljust(6, '0')
        mean = pad(mean)
        std = pad(std)
        if ttest == '=':
            x = '\\textbf{'+ mean + '} ' + ' '.join([pm, std])
        elif ttest == '**':
            x = mean + '$^{\ddag}$ ' + ' '.join([pm, std])
        elif ttest == '*':
            x = mean + '$^{\dag}$ ' + ' '.join([pm, std])
        else:
            x = mean + '$^{\phantom{\dag}}$ ' + ' '.join([pm, std])
        return x

    for dataset in datasets:
        df[dataset] = df[dataset].apply(nice_str)

    return df


def generate_latex_table(df, path, metric):
    pv = df.pivot_table(
        index=['vectors', 'learner', 'method'], 
        columns=['dataset'], 
        values=metric, 
        aggfunc=[np.mean, np.std, np.size]
    )

    pv = pv.reset_index()

    sparse = pv[pv['vectors'] == 'Sparse']

    df_tfidf_lr = run_ttest_in_table(sparse[sparse['learner']=='LR'])
    df_tfidf_svm = run_ttest_in_table(sparse[sparse['learner']=='SVM+LRI'])

    df = pd.concat([df_tfidf_svm, df_tfidf_lr])

    print(df)

    dflatex = df.to_latex(index=False, bold_rows=True, escape = False)
    print('saving to path:', path)
    print(dflatex)

    open(path, 'wt').write(dflatex)


dfs = []
for learner_log, learner, vectors in [
    ('log-aa-svm-r1', 'SVM+LRI', 'Sparse'),
    ('log', 'LR', 'Sparse')
    ]:
    for dataset in ['pan2011', 'victorian', 'imdb62', 'arxiv']:
        for result in glob(f'../{learner_log}/results_{dataset}*.csv'):
            print(f'concatenating: {result}')
            df = pd.read_csv(result, sep='\t')
            df['learner'] = learner
            df['vectors'] = vectors
            dfs.append(df)
df = pd.concat(dfs)

df = df[ df['method'].isin(['PairLRlinear', 'LR']) ]
df = df[ df['n_authors'].isin([25,-1, 100])]
df = df[ df['docs_by_author'].isin([50,-1])]

df.method = df.method.apply(lambda x: x.replace('PairLRlinear', 'DV'))
df.method = df.method.apply(lambda x: x.replace('PairLRknn', 'DV-knn'))
df.method = df.method.apply(lambda x: x.replace('LR', 'STD'))


metric = 'f1_attr_macro'
generate_latex_table(df, f'../latex/add_exp_{metric}-r1.tex', metric=metric)

metric='f1_attr_micro'
generate_latex_table(df, f'../latex/add_exp_{metric}-r1.tex', metric=metric)


