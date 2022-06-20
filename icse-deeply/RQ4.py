import pandas as pd
import os
import re
import seaborn as sns
import random
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from io import StringIO

sns.set()

corr = {
    'jpetstore': 0.85,
    'plants': 0.77,
    'daytrader': 0.97,
    'acmeair': 0.76
}

header = '  Partitions'

_acmeair = []
_daytrader = []
_jpetstore = []
_plants = []

for file in os.listdir('./additional-files/rq4/'):
    if not file.endswith('.out'):
        continue
    
    with open(file, 'r') as f:
        lines = f.readlines()
        
    [acmeair, daytrader, jpetstore, plants] = [lines[i:i+101] for i in range(len(lines)) if lines[i].startswith(header)]
    df = pd.read_table(StringIO('\n'.join(acmeair)), engine='python', sep='\s*\t\s*')
    df['Loss[-]'] = df['BCS[-]'] + df['ICP[-]'] - 1. / corr['acmeair'] * df['SM[+]'] + 0.2 * df['NED[-]']
    _acmeair.append(df[df['Loss[-]'] == df['Loss[-]'].min()].iloc[0,:])
    
    df = pd.read_table(StringIO('\n'.join(daytrader)), engine='python', sep='\s*\t\s*')
    df['Loss[-]'] = df['BCS[-]'] + df['ICP[-]'] - 1. / corr['daytrader'] * df['SM[+]'] + 0.2 * df['NED[-]']
    _daytrader.append(df[df['Loss[-]'] == df['Loss[-]'].min()].iloc[0,:])
    
    df = pd.read_table(StringIO('\n'.join(jpetstore)), engine='python', sep='\s*\t\s*')
    df['Loss[-]'] = df['BCS[-]'] + df['ICP[-]'] - 1. / corr['jpetstore'] * df['SM[+]'] + 0.2 * df['NED[-]']
    _jpetstore.append(df[df['Loss[-]'] == df['Loss[-]'].min()].iloc[0,:])
    
    df = pd.read_table(StringIO('\n'.join(plants)), engine='python', sep='\s*\t\s*')
    df['Loss[-]'] = df['BCS[-]'] + df['ICP[-]'] - 1. / corr['plants'] * df['SM[+]'] + 0.2 * df['NED[-]']
    _plants.append(df[df['Loss[-]'] == df['Loss[-]'].min()].iloc[0,:])


def fix(x):
    """
    Gets the string form of the list (in the output files) to a form that can be parsed.
    """
    return re.sub('\s+',
                  ',',
                  re.sub('\[\s+',
                         '[',
                         x
                  )
           )


_acmeair = pd.concat(_acmeair, axis=1).T
_jpetstore = pd.concat(_jpetstore, axis=1).T
_daytrader = pd.concat(_daytrader, axis=1).T
_plants = pd.concat(_plants, axis=1).T

acmeair_sizes = [y * 100 / sum(eval(fix(x))) for x in _acmeair['ClassSizes'] for y in eval(fix(x))]
jpetstore_sizes = [y * 100 / sum(eval(fix(x))) for x in _jpetstore['ClassSizes'] for y in eval(fix(x))]
plants_sizes = [y * 100 / sum(eval(fix(x))) for x in _plants['ClassSizes'] for y in eval(fix(x))]
daytrader_sizes = [y * 100 / sum(eval(fix(x))) for x in _daytrader['ClassSizes'] for y in eval(fix(x))]

min_size = min(len(acmeair_sizes), len(daytrader_sizes), len(jpetstore_sizes), len(plants_sizes))

random.shuffle(acmeair_sizes)
random.shuffle(daytrader_sizes)
random.shuffle(plants_sizes)
random.shuffle(jpetstore_sizes)

sizes = {
    'acmeair': acmeair_sizes[:min_size],
    'jpetstore': jpetstore_sizes[:min_size],
    'daytrader': daytrader_sizes[:min_size],
    'plants': plants_sizes[:min_size]
}
df = pd.DataFrame(sizes)

fig, ax = plt.subplots(dpi=150)
ax.set_xlabel('Dataset')
ax.set_ylabel('size%')

plot = sns.boxplot(data=df);

medians = df.median().round(2)
means = df.mean().round(2)
vertical_offset = df.median() * 0.06
for xtick in plot.get_xticks():
    plot.text(x=xtick, 
              y=medians[df.columns[xtick]] + vertical_offset[df.columns[xtick]], 
              s=r'med='+f'{medians[df.columns[xtick]]}', 
              horizontalalignment='center', size='small', color='w', weight='semibold')