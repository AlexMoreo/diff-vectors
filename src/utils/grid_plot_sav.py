import seaborn as sns; sns.set(style="ticks", color_codes=True)
import matplotlib.pyplot as plt
from glob import glob
import pandas as pd
import os
from tqdm import tqdm

logpath = '../log-sav'
plotpath = '../plot-sav-acc'

filter_suffix=['_', '_arxiv']

dfs = []
for result in glob('../log-sav/SAV_*.csv'):
# for result in glob('../log-sav/SAV_dist.csv'):
    read_it = True
    for suff in filter_suffix:
        if result.endswith(f'{suff}.csv'):
            read_it=False
    if not read_it: continue
    print(f'loading {result}')
    df = pd.read_csv(result,sep='\t')
    dfs.append(df)
df = pd.concat(dfs)

df.method = df.method.apply(lambda x: x.replace('LR-close', 'StdVectors-Attr'))
df.method = df.method.apply(lambda x: x.replace('LRPairknn-close', 'DiffVectors-SAV'))
df.method = df.method.apply(lambda x: x.replace('LRPairknn-att-close', 'DiffVectors-Attr'))
df.method = df.method.apply(lambda x: x.replace('LRPairknn-open', 'DiffVectors'))
df.method = df.method.apply(lambda x: x.replace('Impostors-open', 'Impostors'))
df.method = df.method.apply(lambda x: x.replace('SVM-close', 'StdVectors-Attr-SVM'))
df.method = df.method.apply(lambda x: x.replace('SVMPairknn-close', 'DiffVectors-SAV-SVM'))
df.method = df.method.apply(lambda x: x.replace('SVMPairknn-att-close', 'DiffVectors-Attr-SVM'))
df.method = df.method.apply(lambda x: x.replace('SVMPairknn-open', 'DiffVectors-SVM'))
df.method = df.method.apply(lambda x: x.replace('-close', ''))
df.method = df.method.apply(lambda x: x.replace('-open', ''))

df = df[df.method != 'Dist-l2']
df = df[df.method != 'Dist-l1']
# df = df[df.method != 'Dist-cos']

print(pd.np.unique(df.method))

print(df.columns)

df_close = df.loc[df['mode']=='CLOSE'].pivot_table(index=['dataset', 'n_authors', 'docs_by_author', 'method'], values=['acc', 'p','r','f1'], aggfunc=lambda x:(x.mean(), x.std())).reset_index()
df_open = df.loc[df['mode']=='OPEN'].pivot_table(index=['dataset', 'n_authors', 'docs_by_author', 'method'], values=['acc', 'p','r','f1'], aggfunc=lambda x:(x.mean(), x.std())).reset_index()

os.makedirs(plotpath) if not os.path.exists(plotpath) else None

sns.set()

def quantile_plot(x, y, **kwargs):
    assert (x.index == y.index).all(), 'wrong'
    x = x.values
    y_mean, y_std = list(zip(*y.values))
    y_mean = pd.np.asarray(y_mean)
    y_std = pd.np.asarray(y_std)
    plt.plot(x, y_mean, **kwargs)
    plt.fill_between(x, y_mean - y_std, y_mean + y_std, alpha=0.25)

def dofacetgrid(df, values, saveto, hue_order):
    g = sns.FacetGrid(df, col="dataset", row="n_authors", hue='method', margin_titles=True, hue_order=hue_order)
    # g = sns.FacetGrid(df, col="n_authors", row="dataset", hue='method', margin_titles=True)
    g = g.map(quantile_plot, "docs_by_author", values, marker='o').add_legend()
    g.savefig(saveto)

plots_request = ['acc']

plot_format='pdf'

for plotvalue in tqdm(plots_request, desc='plotting mode==close'):
    dofacetgrid(df_close, plotvalue, saveto=f'{plotpath}/{plotvalue}-close.{plot_format}', hue_order=['DiffVectors-SAV', 'DiffVectors-SAV-SVM'])

for plotvalue in tqdm(plots_request, desc='plotting mode==open'):
    dofacetgrid(df_open, plotvalue, saveto=f'{plotpath}/{plotvalue}-open.{plot_format}', hue_order=['Dist-cos', 'DiffVectors', 'DiffVectors-SVM', 'Impostors'])


#for plotvalue in tqdm(plots_request, desc='plotting mode==close'):
#    dofacetgrid(df_close, plotvalue, saveto=f'{plotpath}/{plotvalue}-close.{plot_format}', hue_order=['Dist-cos', 'DiffVectors-SAV', 'DiffVectors-Attr', 'StdVectors-Attr'])

#for plotvalue in tqdm(plots_request, desc='plotting mode==open'):
#    dofacetgrid(df_open, plotvalue, saveto=f'{plotpath}/{plotvalue}-open.{plot_format}', hue_order=['Dist-cos', 'DiffVectors', 'Impostors'])

print('done!')

