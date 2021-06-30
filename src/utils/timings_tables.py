import pandas as pd
import numpy as np
from glob import glob

logattr = '../log'

filter_suffix=['tmp']
dfs = []
for result in glob(f'{logattr}/results_*.csv'):
    read_it = True
    for suff in filter_suffix:
        if result.endswith(f'{suff}.csv'):
            read_it=False
    if not read_it: continue
    df = pd.read_csv(result,sep='\t')
    dataset = df.dataset.unique()[0]
    if dataset != 'arxiv':
        df = df[df.n_authors==20]
        df = df[df.docs_by_author == 50]
    dfs.append(df)
df = pd.concat(dfs)

print('--------------')
print('Training documents')
pv = df.pivot_table(
    index='method',
    columns='dataset',
    values='trainsize',
    aggfunc=np.mean
)
print(pv)


df=df.drop(columns=['seed','n_authors','docs_by_author','time_vec','acc_attr','f1_attr_macro','f1_attr_micro','acc_verif','f1_verif', 'trainsize'])

#print(df.columns)
#df.drop(columns=[''])



df2 = pd.read_csv('../timings/clock.csv', sep='\t')

def mean_std(x):
    m=x.mean()
    std=x.std()
    #return f'{m:4.1f}s +-{std:3.1f}'
    return f'{m:4.1f}s'


def tolatex(df, fout, caption="",label="timings:"):

    df.method = df.method.apply(lambda x: x.replace('PairLR', 'DiffVectors-'))
    df.method = df.method.apply(lambda x: x.replace('LRbin', 'StdVectors-Verif'))
    df.method = df.method.apply(lambda x: x.replace('LR', 'StdVectors-Attr'))
    df.method = df.method.apply(lambda x: x.replace('f1_verif', 'f1'))

    df.dataset = df.dataset.apply(lambda x: x.replace('imdb62', '0.imdb62'))
    df.dataset = df.dataset.apply(lambda x: x.replace('pan2011', '1.pan2011'))
    df.dataset = df.dataset.apply(lambda x: x.replace('victorian', '2.victorian'))
    df.dataset = df.dataset.apply(lambda x: x.replace('arxiv', '3.arxiv'))

    df = df.rename(columns={"time_tr": "0.Train", "time_te": "1.Test"})

    pv = df.pivot_table(
        index=['method'],
        columns='dataset',
        values=['0.Train', '1.Test'],
        aggfunc=mean_std
    )

    pv.columns = pv.columns.swaplevel(0, 1)
    pv.sortlevel(0, axis=1, inplace=True)

    #pv.columns.reorder_levels([])

    #column_order = ['time_tr', 'time_te']
    #column_order = [['imdb62', 'pan2011', 'victorian', 'arxiv'],['time_tr','time_te']]
    #column_order = pd.MultiIndex.from_product([['imdb62', 'pan2011', 'victorian', 'arxiv'],['time_tr','time_te']])
    #pv = pv.reindex(column_order, 1)

    print(pv)

    latex = pv.to_latex()
    latex = latex.replace('+-','$\pm$')
    latex = latex.replace('0.imdb62', 'imdb62')
    latex = latex.replace('1.pan2011', 'pan2011')
    latex = latex.replace('2.victorian', 'victorian')
    latex = latex.replace('3.arxiv', 'arxiv')
    latex = latex.replace('0.Train', 'Train')
    latex = latex.replace('1.Test', 'Test')
    latex = "\\begin{table}[ht!]\n\center\n" + latex + '\\caption{'+caption+'}\n\\label{'+label+'}\n\end{table}\n'
    print(latex)
    open(fout, 'wt').write(latex)

tolatex(df, '../latex/timings_attr.tex', caption='Training and testing times clocked in AA', label='timings:attr')
tolatex(df2, '../latex/timings_sav.tex', caption='Training and testing times clocked in SAV', label='timings:sav')



