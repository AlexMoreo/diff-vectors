import numpy as np
import pandas as pd
from glob import glob
from utils.common import ttest


if __name__ == '__main__':
    logpath = '../log'

    dfs = []
    for result in glob('../log/results_arxiv*.csv'):
        df = pd.read_csv(result, sep='\t')
        dfs.append(df)
    df = pd.concat(dfs)

    df.method = df.method.apply(lambda x: x.replace('PairLRknn', 'Lazy AA'))
    df.method = df.method.apply(lambda x: x.replace('PairLRlinear', 'Stacked AA'))
    df.method = df.method.apply(lambda x: x.replace('LRbin', 'STD-Bin'))
    df.method = df.method.apply(lambda x: x.replace('LR', 'STD-AA'))


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

    metric='f1_verif'
    generate_latex_table(df, f'../latex/attr_arxiv_{metric}-r1.tex', metric=metric)

    # metric='acc_verif'
    # generate_latex_table(df, f'../latex/attr_arxiv_{metric}-r1.tex', metric=metric)


