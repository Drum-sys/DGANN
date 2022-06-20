from __future__ import division
from __future__ import print_function

import argparse
from ast import arg
import time
import sys
import os
import json
import pickle
from tkinter import Menubutton
from traceback import print_tb
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch import optim
from matplotlib import pyplot as plt

from model import GCNAE, GAT
from optimizer import compute_structure_loss, compute_attribute_loss, update_o1, update_o2
from utils import load_data_cma, preprocess_graph, plot_losses
from kmeans import Clustering
from metrics.metrics_util import *

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_vae', help="models used")
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
# 5000 epochs is where the pretrain loss starts hovering around 5.0 - 6.0
parser.add_argument('--k', type=int, default=6, help='Number of clusters.')
parser.add_argument('--preepochs', type=int, default=5000, help='Number of epochs to pre-train.')
parser.add_argument('--clusepochs', type=int, default=1, help='Number of epochs to pre-train for clustering.')
parser.add_argument('--epochs', type=int, default=600, help='Number of epochs to train.')
parser.add_argument('--hidden1', type=int, default=64, help='Number of units in hidden layer 1.')
parser.add_argument('--hidden2', type=int, default=32, help='Number of units in hidden layer 2.')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--lambda1', type=float, default=0.1, help='Structure loss weight.')
parser.add_argument('--lambda2', type=float, default=0.1, help='Attribute loss weight.')
parser.add_argument('--lambda3', type=float, default=0.8, help='Clustering loss weight.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='leakeyRelu')
parser.add_argument('--dataset_str', type=str, default=None, help='type of dataset.')
parser.add_argument('--outputdir', type=str, default=None, help='type of outdir.')
parser.add_argument('--outfile', type=str, default='embeddings', help='output embeddings file.')

args = parser.parse_args()

def gae_for(args):
    print("Using {} dataset".format(args.dataset_str))
    adj, features, adj_tensor = load_data_cma(args.dataset_str)
    n_nodes, feat_dim = features.shape

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    model = GCNAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    # model = GAT(feat_dim, args.hidden2, args.alpha, args.dropout)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    gamma = 0.98
    schedule_update_interval = 200
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # initialize all outlier scores with the equal values summing to 1
    init_value = [1./n_nodes] * n_nodes
    o_1 = torch.FloatTensor(init_value) # structural outlier
    o_2 = torch.FloatTensor(init_value) # attribute outlier

    lossfn = nn.MSELoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    lambda1 = args.lambda1 / (args.lambda1 + args.lambda2)
    lambda2 = args.lambda2 / (args.lambda1 + args.lambda2)

    kmeans = Clustering(args.k)

    # PRETRAIN ON STRUCTURE AND ATTRIBUTE LOSSES, NO OUTLIER LOSS
    for epoch in range(args.preepochs):
        model.train()
        optimizer.zero_grad()

        # GCN
        recon, embed = model(features, adj_norm)
        
        # GAT
        # recon, embed = model(features, adj_tensor)

        structure_loss = compute_structure_loss(adj_norm, embed, o_1)
        attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)

        loss = lambda1 * structure_loss + lambda2 * attribute_loss

        # Update the functions F and G (embedding network)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "time=", "{:.5f}".format(time.time() - t), "Pretrain:",pretrain)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "lr=", "{:.5f}".format(scheduler.get_last_lr()[0]))

    # Initialize clusters
    recon, embed = model(features, adj_norm)
    
    # GAT
    # recon, embed = model(features, adj_tensor)
    kmeans.cluster(embed)

    # TRAIN ON ALL THREE LOSES WITH OUTLIER UPDATES
    for epoch in range(args.epochs):
        # Update the values of O_i1 and O_i2
        o_1 = update_o1(adj_norm, embed)
        o_2 = update_o2(features, recon)

        if (epoch+1) % schedule_update_interval == 0:
            scheduler.step()

        model.train()
        optimizer.zero_grad()
        
        # GCN
        recon, embed = model(features, adj_norm)
        
        # GAT
        # recon, embed = model(features, adj_tensor)

        kmeans.cluster(embed)

        structure_loss = compute_structure_loss(adj_norm, embed, o_1)
        attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)
        clustering_loss = kmeans.get_loss(embed)

        loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss) + (args.lambda3 * clustering_loss)
        #loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss)

        # Update the functions F and G (embedding network)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "time=", "{:.5f}".format(time.time() - t), "Pretrain:",pretrain)
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(cur_loss), "lr=", "{:.5f}".format(scheduler.get_last_lr()[0]))

    # Extract embeddings
    adj_norm = preprocess_graph(adj)
    recon, embed = model(features, adj_norm)
    # GAT
    # recon, embed = model(features, adj_tensor)
    embed = embed.detach().cpu().numpy()

    embfile = os.path.join(args.outputdir, args.outfile+".pkl")
    with open(embfile,"wb") as f:
        pickle.dump(embed, f)

    o_1 = o_1.detach().cpu().numpy()
    o_2 = o_2.detach().cpu().numpy()
    outlfile = os.path.join(args.outputdir, args.outfile+"_outliers.pkl")
    with open(outlfile,"wb") as f:
        pickle.dump([o_1, o_2], f)

    membfile = os.path.join(args.outputdir, args.outfile+"_membership.pkl")
    with open(membfile,"wb") as f:
        pickle.dump(kmeans.get_membership(), f)
    
    member = kmeans.get_membership()
    print("member:", member)
    
    mapping_path = os.path.join(args.dataset_str, "mapping.json")
    print("mapping_path:", mapping_path)
    with open(mapping_path) as mapping_file:
        mapping = json.load(mapping_file)
        
    vertical_cluster_assignment = {class_name: int(
                    member[int(id_)]) for id_, class_name in mapping.items()}
    print(vertical_cluster_assignment)
    
    with open(Path(args.outputdir).joinpath('vertical_cluster_assignment_{}.json'.format(args.k)), 'w') as cluster_assgn_file:
                json.dump(vertical_cluster_assignment,
                          cluster_assgn_file, indent=4)

if __name__ == '__main__':
    gae_for(args)
    PROJECT_DIR = "C:\\python\\cogcn\\cogcn\\datasets_runtime1\\"
    
    #app name
    app = "acmeair"
    benchmark_type = "cogcn"
    ROOT = 'Root' #$Root$
    
    #from mono2micro output dir get partition file, bcs_per_class, runtime_call_volume
    OUTPUT_DIR = os.path.join(PROJECT_DIR, app, benchmark_type+"_output", "GCN")
    

    bcs_per_class = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "bcs_per_class.json"), 'r') as f:
        bcs_per_class = json.load(f)
    
    runtime_call_volume = {}
    with open(os.path.join(PROJECT_DIR, app, "mono2micro_output", "runtime_call_volume.json"), 'r') as f:
        runtime_call_volume = json.load(f)
   
    partition_sizes = [args.k]  
    for k in partition_sizes:
        with open(os.path.join(OUTPUT_DIR, "vertical_cluster_assignment_"+str(k)+".json"), 'r') as f:
            partition = json.load(f)
        class_bcs_partition_assignment, partition_class_bcs_assignment = gen_class_assignment(partition, bcs_per_class)
        # bcs = business_context_purity(partition_class_bcs_assignment)
        # icp = inter_call_percentage(ROOT, class_bcs_partition_assignment, runtime_call_volume)
        sm = structural_modularity(partition_class_bcs_assignment,runtime_call_volume)
        #mq = metrics_util.modularity_quality(ROOT, partition, runtime_call_volume)
        mq = modular_quality(ROOT, partition_class_bcs_assignment,runtime_call_volume)
        ifn = interface_number(ROOT, partition_class_bcs_assignment,runtime_call_volume)
        ned = non_extreme_distribution(partition_class_bcs_assignment)
        
        # print(str(k)+","+str(bcs)+","+str(icp)+","+str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))
        print(str(sm)+","+str(mq)+","+str(ifn)+","+str(ned))
