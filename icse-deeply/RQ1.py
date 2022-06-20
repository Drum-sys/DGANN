from importlib.resources import path
from statistics import median
from traceback import print_tb
from raise_utils.interpret.sk import Rx
from pathlib import Path
from io import StringIO
import pandas as pd
import json
import os
import sys


sys.path.append('./Bunch/output')
from metrics.metrics_util import *


datasets = ['jpetstore', 'plants', 'daytrader', 'acmeair']

def compute_metrics(app, clusters_assign, num_part) -> list:
    """
    Computes metrics for a given app and cluster assignments, where
    `num_part` is the number of partitions.
    """
    # Some files required to compute metrics
    app_path = Path('./additional-files/json/')
    with open(app_path.joinpath(app, "bcs_per_class.json"), 'r') as f:
        bcs_per_class = json.load(f)

    with open(app_path.joinpath(app, "runtime_call_volume.json"), 'r') as f:
        runtime_call_volume = json.load(f)

    # Compute Metrics
    ROOT = 'Root'
    class_bcs_partition_assignment, partition_class_bcs_assignment = gen_class_assignment(
        clusters_assign, bcs_per_class)
    bcs = business_context_purity(partition_class_bcs_assignment)
    icp = inter_call_percentage(
        ROOT, class_bcs_partition_assignment, runtime_call_volume)
    sm = structural_modularity(
        partition_class_bcs_assignment, runtime_call_volume)
    mq = modular_quality(
        ROOT, partition_class_bcs_assignment, runtime_call_volume)
    ifn = interface_number(
        ROOT, partition_class_bcs_assignment, runtime_call_volume)
    loss = bcs + icp - sm - mq + ifn

    results = [num_part, bcs, icp, sm, mq, ifn, loss]
    return results


#########
# Bunch #
#########
def get_bunch_clusters(app):
    filename = f'./additional-files/output/{app}/bunch.txt'
    with open(filename, 'r') as f:
        lines = f.readlines()

    assign_dicts = []
    cur_dict = {}
    cur_n_clusters = 0
    run = -1

    for line in lines:
        if len(line) < 5:
            run += 1

            if run != 0 and cur_dict != {}:
                assign_dicts.append(cur_dict)
                cur_dict = {}
                cur_n_clusters = 0
            continue

        for tok in line.split():
            cur_dict[tok] = cur_n_clusters

        cur_n_clusters += 1

    assign_dicts.append(cur_dict)
    return assign_dicts


def get_bunch_results(dataset: str):
    clusters = get_bunch_clusters(dataset)
    results = []
    for assign in clusters:
        n_clusters = max(assign.values())
        results.append(compute_metrics(dataset, assign, n_clusters)[1:4])

    return pd.DataFrame(results, columns=['BCS [-]', 'ICP [-]', 'SM [+]'])


#########
# FoSCI #
#########
def parse_fosci_file(filename: str, batch_size=100, agg_by='   Loss[-] ', agg_fn=min):
    """
    Parses the result of a tuning output file.

    :param {str} filename - The filename
    :param {str} batch_size - The number of iters the algorithm was run for
    :param {str} agg_by - The column to aggregate by
    :param {str} agg_fn - Aggregation function. Will be passed df[agg_by].
    """
    results = pd.read_csv(filename, sep='|')

    try:
        results.drop(results.columns[[0, 1, -1]], axis=1, inplace=True)
        results.columns = [x.strip() for x in results.columns]
        results[agg_by] = results['BCS [-]'] + results['ICP [-]'] - results['SM [+]']
        tuned_df_dict = {n: results.iloc[n:n+batch_size, :]
                         for n in range(0, len(results), batch_size)}

        tuned_df = [x[x[agg_by] == agg_fn(x[agg_by])].iloc[0,:].to_frame().T for x in tuned_df_dict.values()]
        return pd.concat(tuned_df, ignore_index=True)
    except KeyError as e:
        print('KeyError occurred:', e)
        print(results.columns)


def get_fosci_results(dataset: str):
    filename = f'./additional-files/output/{dataset}/fosci.txt'
    return parse_fosci_file(filename)


##########
# CO-GCN #
##########
def parse_cogcn_file(filename: str, batch_size=60, agg_by='MQ [+]', agg_fn=max):
    """
    Parses the result of a tuning output file.

    :param {str} filename - The filename
    :param {str} batch_size - The number of iters the algorithm was run for
    :param {str} agg_by - The column to aggregate by
    :param {str} agg_fn - Aggregation function. Will be passed df[agg_by].
    """
    tuned = pd.read_csv(filename, sep='\t')
    tuned.columns = [x.strip() for x in tuned.columns]
    tuned.columns = [x.split('[')[0] + ' [' + x.split('[')[1] if '[' in x else x for x in tuned.columns]

    try:
        tuned.drop(['Partitions', 'WC_time [-]'], axis=1, inplace=True)
        tuned_df_dict = {n: tuned.iloc[n:n+batch_size, :]
                         for n in range(0, len(tuned), batch_size)}

        tuned_df = [x[x[agg_by] == agg_fn(x[agg_by])] for x in tuned_df_dict.values()]
        return pd.concat(tuned_df)
    except KeyError as e:
        print('KeyError occurred:', e)
        print(tuned.columns)


def get_cogcn_results(dataset: str):
    filename = f'./additional-files/output/{dataset}/cogcn.txt'
    return parse_cogcn_file(filename)


#######
# MEM #
#######
def parse_mem_file(filename: str, batch_size=100, agg_by='Loss [-]', agg_fn=min):
    """
    Parses the result of a tuning output file.

    :param {str} filename - The filename
    :param {str} batch_size - The number of iters the algorithm was run for
    :param {str} agg_by - The column to aggregate by
    :param {str} agg_fn - Aggregation function. Will be passed df[agg_by].
    """
    tuned = pd.read_csv(filename, sep='|')
    tuned.columns = [x.strip() for x in tuned.columns]
    tuned.columns = [x.split('[')[0] + ' [' + x.split('[')[1] if '[' in x else x for x in tuned.columns]

    try:
        tuned.drop(tuned.columns[[0, 1, 2, -1, -3]], axis=1, inplace=True)
        tuned_df_dict = {n: tuned.iloc[n:n+batch_size, :]
                         for n in range(0, len(tuned), batch_size)}

        tuned_df = [x[x[agg_by] == agg_fn(x[agg_by])].iloc[0,:].to_frame().T for x in tuned_df_dict.values()]
        return pd.concat(tuned_df, ignore_index=True)
    except KeyError as e:
        print('KeyError occurred:', e)
        print(tuned.columns)


def get_mem_results(dataset: str):
    filename = f'./additional-files/output/{dataset}/mem.txt'
    return parse_mem_file(filename)


##############
# mono2micro #
##############
def get_mono2micro_results(dataset: str):
    with open('./additional-files/output/mono2micro.txt', 'r') as f:
        lines = f.readlines()

    lines = filter(lambda p: p.startswith('|'), lines)
    lines = map(lambda x: x[1:-2], lines)
    lines = list(filter(lambda p: '[-]' not in p and '---' not in p, lines))

    df = pd.read_csv(StringIO('\n'.join(lines)), sep='|', dtype=float,
                     header=None, names=['k', 'BCS [-]', 'ICP [-]', 'SM [+]', 'MQ [+]', 'IFN [-]', 'Time'])

    data_idx = [y for x in [range(i, i+6) for i in range(datasets.index(dataset), 720, 24)] for y in x]
    data = df.iloc[data_idx, :]
    data.loc[:,'Loss'] = data['BCS [-]'] + data['ICP [-]'] - data['SM [+]']

    data = data.reset_index()
    res = data.groupby(data.index // 6).agg({'BCS [-]': 'last', 'ICP [-]': 'last', 'SM [+]': 'last', 'Loss': 'min'})
    return res


##########
# DEEPLY #
##########
def get_our_results(dataset: str):
    # path = './additional-files/output/deeply/'
    path = "C:\\python\\icse-deeply\\additional-files\\output\\deeply\\"
    header = '  Partitions'

    results = {x: [] for x in datasets}

    for file in os.listdir(path):
        if not file.endswith('.out'):
            continue

        with open(path + file, 'r') as f:
            lines = f.readlines()

        _r = {x: [] for x in datasets}
        [_r['acmeair'], _r['daytrader'], _r['jpetstore'], _r['plants']] = [lines[i:i+101] for i in range(len(lines)) if lines[i].startswith(header)]
        df = pd.read_table(StringIO('\n'.join(_r[dataset])), sep=r'\s+', dtype=float)
        df.columns = [x.strip() for x in df.columns]
        df.columns = [x.split('[')[0] + ' [' + x.split('[')[1] if '[' in x else x for x in df.columns]

        # WC_time[-] is actually Loss
        results[dataset].append(df[df['WC_time [-]'] == df['WC_time [-]'].min()].iloc[0,1:4])

    return pd.concat(results[dataset], axis=1).T


###################
# DEEPLY-w/o loss #
###################
def get_our_lossless_results(dataset: str):
    path = './additional-files/output/deeply-lossless/'
    header = '  Partitions'

    results = {x: [] for x in datasets}

    for file in os.listdir(path):
        if not file.endswith('.out'):
            continue

        with open(path + file, 'r') as f:
            lines = f.readlines()

        _r = {x: [] for x in datasets}
        [_r['acmeair'], _r['daytrader'], _r['jpetstore'], _r['plants']] = [lines[i:i+101] for i in range(len(lines)) if lines[i].startswith(header)]
        df = pd.read_table(StringIO('\n'.join(_r[dataset])), engine='python', sep=r'\s*\t\s*')
        df.columns = [x.strip() for x in df.columns]
        df.columns = [x.split('[')[0] + ' [' + x.split('[')[1] if '[' in x else x for x in df.columns]
        df.drop('ClassSizes', axis=1)

        # WC_time[-] is actually Loss
        results[dataset].append(df[df['Loss [-]'] == df['Loss [-]'].min()].iloc[0,1:])

    return pd.concat(results[dataset], axis=1).T




# ###################
# # DEEPLY-GAT loss
# ###################
# def get_our_lossless_results(dataset: str):
#     path = './additional-files/output/deeply-lossless/GAT'
#     results = {x: [] for x in datasets}

#     for file in os.listdir(path):
#         if not file.endswith('.out'):
#             continue

#         with open(path + file, 'r') as f:
#             lines = f.readlines()

#         _r = {x: [] for x in datasets}
#         [_r['acmeair'], _r['daytrader'], _r['jpetstore'], _r['plants']] = [lines[i:i+101] for i in range(len(lines)) if lines[i].startswith(header)]
#         df = pd.read_table(StringIO('\n'.join(_r[dataset])), engine='python', sep=r'\s*\t\s*')
#         df.columns = [x.strip() for x in df.columns]
#         df.columns = [x.split('[')[0] + ' [' + x.split('[')[1] if '[' in x else x for x in df.columns]
#         df.drop('ClassSizes', axis=1)

#         # WC_time[-] is actually Loss
#         results[dataset].append(df[df['Loss [-]'] == df['Loss [-]'].min()].iloc[0,1:4])

#     return pd.concat(results[dataset], axis=1).T



# Statistics
all_results = {
    # 'bunch': { x: get_bunch_results(x) for x in datasets },
    # 'fosci': { x: get_fosci_results(x) for x in datasets },
    # 'cogcn': { x: get_cogcn_results(x) for x in datasets },
    # 'mem': { x: get_mem_results(x) for x in datasets },
    # 'mono2micro': { x: get_mono2micro_results(x) for x in datasets },
    'us': { x: get_our_results(x) for x in datasets },
    # 'us-loss': { x: get_our_lossless_results(x) for x in datasets }
}
# for dataset in datasets:
#     print(dataset)
#     print('=' * len(dataset))
#     print()
#     for metric in all_results['us']['jpetstore'].columns:
#         print(metric)
#         print('-' * len(metric))
#         compare_dict = { key: all_results[key][dataset][metric] for key in all_results }
#         Rx.show(Rx.sk(Rx.data(**compare_dict)))
#         print()



data = get_our_lossless_results("acmeair")
print(data)
print("\n", len(data))
for metric in data.columns:
    value = data[metric].values
    median = np.median(value)
    print("metric:{}, median:{}\n".format(metric, median) )
# for x in datasets:
#     data = get_our_lossless_results(x)
#     print("{}\n{}".format(x, data))