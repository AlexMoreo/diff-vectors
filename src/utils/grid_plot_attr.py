import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os

logpath = '../log'
plotpath = '../plot'

filter_suffix=['tmp', 'LRbin']

dfs = []
for result in glob('../log/results_*.csv'):
    read_it = True
    for suff in filter_suffix:
        if result.endswith(f'{suff}.csv'):
            read_it=False
    if not read_it: continue
    df = pd.read_csv(result,sep='\t')
    # dataset_name = result[result.index('results_')+len('results_'):result.index('-cv')]
    # df['dataset'] = dataset_name
    dfs.append(df)
df = pd.concat(dfs)
df = df[df.dataset!='arxiv']
# df['log(trsize)'] = pd.np.log(df['trainsize'])
df['macro-$F_1$'] = df['f1_verif']
# df.method = df.method.apply(lambda x: x.replace('CLS', 'LR'))
df.method = df.method.apply(lambda x: x.replace('PairLRknn', 'Lazy AA'))
df.method = df.method.apply(lambda x: x.replace('PairLRlinear', 'Stacked AA'))
df.method = df.method.apply(lambda x: x.replace('LRbin', 'STD-Bin'))
df.method = df.method.apply(lambda x: x.replace('LR', 'STD AA'))
#df.method = df.method.apply(lambda x: x.replace('f1_verif', 'f1'))

print(df.columns)

df = df.pivot_table(index=['dataset', 'n_authors', 'docs_by_author', 'method'], values=['trainsize',
       'time_vec', 'time_tr', 'time_te', 'acc_attr', 'f1_attr_macro',
       'f1_attr_micro', 'acc_verif', 'macro-$F_1$'], aggfunc=lambda x:(x.mean(), x.std())).reset_index()

os.makedirs(plotpath) if not os.path.exists(plotpath) else None

sns.set()

def quantile_plot(x, y, **kwargs):
    assert (x.index == y.index).all(), 'wrong'
    x = x.values

    y_mean, y_std = list(zip(*y.values))
    y_mean = pd.np.asarray(y_mean)
    y_std = pd.np.asarray(y_std)
    #y_mean, y_std = pd.np.log(y_mean), pd.np.log(y_std)
    plt.plot(x, y_mean, **kwargs)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25)


def dofacetgrid(df, values, saveto):
    g = sns.FacetGrid(df, col="dataset", row="n_authors", hue='method', margin_titles=True)
    # g = g.map(plt.plot, "docs_by_author", values, marker='o').add_legend()
    g = g.map(quantile_plot, "docs_by_author", values, marker='o').add_legend()
    g.savefig(saveto)

plots_request = ['macro-$F_1$']#, 'f1_attr_macro'] #, 'trainsize', 'time_tr', 'time_te']
#plots_request = ['time_tr', 'time_te']

for plotvalue in plots_request:
    dofacetgrid(df, plotvalue, saveto=f'{plotpath}/{plotvalue}_.pdf')

print('done!')

