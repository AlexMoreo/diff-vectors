import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob



#n_authors	docs_by_author	method	acc_attr	f1_attr	acc_verif	f1_verif

logfiles = sorted(glob.glob("../log/*.csv"))
print(logfiles)

for log in logfiles:
    result = pd.read_csv(log, sep='\t')
    nauthors = result.n_authors.values
    ndocs = result.docs_by_author.values

    x = nauthors*ndocs
    y = result.f1_verif.values

    label = log.replace('../log/results_cv_','').replace('.csv','')
    plt.scatter(x, y, label=label)


    plt.xlabel('#documents')
    plt.ylabel('performance(F1)')
    print(f'media{label}={y.mean()}')
    # plt.title('About as simple as it gets, folks')
    # plt.grid(True)
    # plt.savefig("test.png")

plt.legend()
plt.show()