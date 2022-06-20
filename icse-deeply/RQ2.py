import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


sns.set()

after_data_deeply = np.array([
    [1, 1, 2],
    [2, 1, 2],
    [3, 1, 1],
    [1, 1, 1]
])
after_data_deeply = np.mean(after_data_deeply, axis=0).reshape((1, 3))
after = pd.DataFrame(after_data_deeply.T, index=['BCS', 'ICP', 'SM'], columns=['DEEPLY'])

before_data = np.array([
    [3.25, 4.5, 2.25],
    [2.25, 3.25, 5],
    [5, 2.25, 3],
    [3.25, 4.75, 5],
    [4.25, 4.75, 3.5]
]).T
before = pd.DataFrame(before_data, index=after.index, columns=['Bunch', 'mono2micro', 'CO-GCN', 'FoSCI', 'MEM'])

fig, (ax0, ax1) = plt.subplots(nrows=1, 
                               ncols=2, 
                               figsize=(5,2), 
                               gridspec_kw={
                                   'width_ratios': [4, 2]
                               }, 
                               sharey=True,
                               dpi=150)

sns.heatmap(before, cmap='RdYlGn_r', vmin=1., vmax=5., ax=ax0, square=True, cbar=False);
sns.heatmap(after, cmap='RdYlGn_r', vmin=1., vmax=5., ax=ax1, square=True, cbar=True);

ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0);
ax0.set_xticklabels(ax0.get_xticklabels(), rotation=45);