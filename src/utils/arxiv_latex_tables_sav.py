import numpy as np
import pandas as pd

df = pd.read_csv('../log-sav-r1/SAV_arxiv.csv', sep='\t')
df.method = df.method.apply(lambda x: x.replace('LR-close', 'STD-2xAA'))
df.method = df.method.apply(lambda x: x.replace('LRStdVectors-Attr', 'STD-2xAA'))
df.method = df.method.apply(lambda x: x.replace('Dist-cos', 'STD-CosDist'))
df.method = df.method.apply(lambda x: x.replace('LR-DiffVectors-SAV', 'DV-Bin'))
df.method = df.method.apply(lambda x: x.replace('LRDiffVectors-Attr', 'DV-2xAA'))
# deprecated replacements... still useful for old result logs
df.method = df.method.apply(lambda x: x.replace('LRPairknn-close', 'DiffVectors-SAV'))
df.method = df.method.apply(lambda x: x.replace('LRPairknn-att-close', 'DiffVectors-Attr'))
df.method = df.method.apply(lambda x: x.replace('LRPairknn-open', 'DiffVectors'))
df.method = df.method.apply(lambda x: x.replace('Impostors-open', 'Impostors'))
df.method = df.method.apply(lambda x: x.replace('-close', ''))
df.method = df.method.apply(lambda x: x.replace('-open', ''))
df = df[df.method!='Dist-l1']
df = df[df.method!='Dist-l2']
from scipy.stats import ttest_ind_from_stats

df['acc'] /= 100
df_close = df.loc[df['mode']=='CLOSE']
df_open = df.loc[df['mode']=='OPEN']

print(df_open)

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
    table = pv.to_latex(index=False, bold_rows=True, float_format="{:0.3f}".format)
    open(path, 'wt').write(table)
    print(table)

metric='acc'
generate_latex_table(df_close, f'../latex/sav_arxiv_close_{metric}-r1.tex', metric=metric)
generate_latex_table(df_open, f'../latex/sav_arxiv_open_{metric}-r1.tex', metric=metric)