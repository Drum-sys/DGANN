from __future__ import division
from __future__ import print_function

import os
import pickle

import torch
import torch.nn as nn
from torch import optim

from model import Encoder, GAT
from optimizer import compute_structure_loss, compute_attribute_loss, update_o1, update_o2
from spectral import Clustering
from utils import load_data_cma, preprocess_graph


def train(args, num_clusters, datapath):
    if args.verbose == True:
        print("Using {} dataset".format(datapath))
    
    adj, features, adj_tensor = load_data_cma(datapath)
    n_nodes, feat_dim = features.shape

    # Some preprocessing
    adj_norm = preprocess_graph(adj)

    #model = GCNAE(feat_dim, args.hidden1, args.hidden2, args.dropout)
    # model = Encoder(feat_dim, args.hidden1, args.hidden2, args.dropout)
    
    #GAT
    model = GAT(feat_dim, args.hidden2, args.alpha, args.dropout)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    schedule_update_interval = 200
    total_steps = args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=total_steps)

    # initialize all outlier scores with the equal values summing to 1
    init_value = [1./n_nodes] * n_nodes
    o_1 = torch.FloatTensor(init_value) # structural outlier
    o_2 = torch.FloatTensor(init_value) # attribute outlier

    lossfn = nn.MSELoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    lambda1 = args.lambda1 / (args.lambda1 + args.lambda2)
    lambda2 = args.lambda2 / (args.lambda1 + args.lambda2)

    cluster = Clustering(num_clusters)

    # PRETRAIN ON STRUCTURE AND ATTRIBUTE LOSSES, NO OUTLIER LOSS
    for epoch in range(args.preepochs):
        model.train()
        optimizer.zero_grad()

        # GCN
        # recon, embed = model(features, adj_norm)
        
        
        # GAT
        recon, embed = model(features, adj_tensor)
        
        structure_loss = compute_structure_loss(adj_norm, embed, o_1)
        attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)

        loss = lambda1 * structure_loss + lambda2 * attribute_loss

        # Update the functions F and G (embedding network)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        if (epoch+1) % 100 == 0 and args.verbose == True:
            if args.verbose:
                print("Epoch:", '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(cur_loss),
                      "lr=", "{:.5f}".format(scheduler.get_last_lr()[0]))

    # Initialize clusters
    # recon, embed = model(features, adj_norm)
    
    # GAT
    recon, embed = model(features, adj_tensor)
    
    cluster.cluster(embed)

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
        # recon, embed = model(features, adj_norm)
        
        # GAT
        recon, embed = model(features, adj_tensor)

        cluster.cluster(embed)

        structure_loss = compute_structure_loss(adj_norm, embed, o_1)
        attribute_loss = compute_attribute_loss(lossfn, features, recon, o_2)
        clustering_loss = cluster.get_loss(embed)

        loss = (args.lambda1 * structure_loss) + (args.lambda2 * attribute_loss) + (args.lambda3 * clustering_loss)
        
        # Update the functions F and G (embedding network)
        loss.backward()
        cur_loss = loss.item()
        optimizer.step()

        if (epoch+1) % 100 == 0:
            if args.verbose:
                print("Epoch:",
                      '%04d' % (epoch + 1),
                      "train_loss=", "{:.5f}".format(cur_loss),
                      "lr=", "{:.5f}".format(scheduler.get_last_lr()[0]))

    # Extract embeddings
    adj_norm = preprocess_graph(adj)
    
    # GCN
    # recon, embed = model(features, adj_norm)
    
    # GAT
    recon, embed = model(features, adj_tensor)
    
    embed = embed.detach().cpu().numpy()

    if args.dumplogs:
        embfile = os.path.join(datapath, args.outfile+".pkl")
        with open(embfile,"wb") as f:
            pickle.dump(embed, f)

    if args.dumplogs:           
        o_1 = o_1.detach().cpu().numpy()
        o_2 = o_2.detach().cpu().numpy()
        outlfile = os.path.join(datapath, args.outfile+"_outliers.pkl")
        with open(outlfile,"wb") as f:
            pickle.dump([o_1, o_2], f)

    memberships = cluster.get_membership()
    membfile = os.path.join(datapath, args.outfile+"_membership.pkl")
    if args.dumplogs:           
        with open(membfile,"wb") as f:
            pickle.dump(memberships, f)

    return memberships
