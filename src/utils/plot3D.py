'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel

#n_authors	docs_by_author	method	acc_attr	f1_attr	acc_verif	f1_verif
file_proposal = '../log/results_cv_PairCLS-k3.csv'
file_base = '../log/results_cv_LR.csv'

df_pairs = pd.read_csv(file_proposal, sep='\t')
df_lr = pd.read_csv(file_base, sep='\t')

n_authors = len(df_pairs.n_authors.unique())
n_docs = len(df_pairs.docs_by_author.unique())
n_authors_lr = len(df_lr.n_authors.unique())
n_docs_lr = len(df_lr.docs_by_author.unique())
assert (n_authors == n_authors_lr) and (n_docs == n_docs_lr), 'experiments do not match!'

X = df_pairs.n_authors.values.reshape(n_authors, n_docs)
Y = df_pairs.docs_by_author.values.reshape(n_authors, n_docs)
Z_pairs = df_pairs.f1_verif.values.reshape(n_authors, n_docs)
Z_lr = df_lr.f1_verif.values.reshape(n_authors, n_docs)
Z = 100*(Z_pairs-Z_lr)

fig = plt.figure()
ax = fig.gca(projection='3d')

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(Z.min(), Z.max())
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

ax.set_xlabel('n_authors')
ax.set_ylabel('docs_by_author')
ax.set_zlabel('performance (f1)')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

Z_pairs = df_pairs.f1_verif.values
Z_lr = df_lr.f1_verif.values

mean_pairs = Z_pairs.mean()
mean_lr = Z_lr.mean()

print(f'mean Pairs = {mean_pairs:.3f} std={Z_pairs.std():.5f}')
print(f'mean LR = {mean_lr:.3f} std={Z_lr.std():.5f}')

_, pval = ttest_rel(Z_pairs, Z_lr)

print(f'p-value={pval:.5f}')

