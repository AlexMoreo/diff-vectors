
import pandas as pd
import numpy as np
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import os

results_C100 = []
results_C1000 = []
results_cv = []


results = []
for file in os.listdir('../log'):
    if 'cv' in file:
        results_cv.append((pd.read_csv(file, sep='\t'), file.rsplit("k", 1)[-1].strip('.csv')))
    elif '1000' in file:
        results_C1000.append((pd.read_csv(file, sep='\t'), file.rsplit("k", 1)[-1].strip('.csv')))
    else:
        results_C100.append((pd.read_csv(file, sep='\t'), file.rsplit("k", 1)[-1].strip('.csv')))
    #df = pd.read_csv(file, sep='\t')
    #name = file.split("_", 1)[1].strip('.csv')
    #results.append((df, name))


def difference (LR, PairCLS):
    return np.array([cls - lr for lr, cls in zip(LR, PairCLS)])


# for x in range(1, len(results[0][0]), 2):
#     n_authors = results[0][0]["n_authors"][x]
#     docs_by_author = results [0][0]["docs_by_author"][x]
#     best_PairCLS = ''
#     best_score = -10
#     for result, name in results:
#         PairCLS = result.ix[x, "f1_verif"]
#         LR = result.ix[(x-1), "f1_verif"]
#         if (LR-PairCLS)>best_score:
#             best_score = (LR-PairCLS)
#             best_PairCLS = name
#     print(f'{n_authors} {docs_by_author} {best_PairCLS} {best_score}')



fig = plt.figure()
x = np.array(results_C100[0].loc[results_C100[0]["method"] == "PairCLS", "n_authors"])
y = np.array(results_C100[0].loc[results_C100[0]["method"] == "PairCLS", "docs_by_author"])
for result, k in results_C100:
    z = difference(result.loc[result["method"] == "LR", "f1_verif"],
                   result.loc[result["method"] == "PairCLS", "f1_verif"])
    ax = fig.gca(projection='3d')
    ax.plot(x,y,z, label=f'k={k}')
    plt.show()

