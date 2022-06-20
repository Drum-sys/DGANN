import sys
import torch
import torch.nn as nn

import numpy as np
from sklearn.cluster import SpectralClustering
from scipy.spatial import KDTree


class Clustering(object):

    def __init__(self, k ):
        self.k = k
        self.assignments = None

    def cluster(self, embed):
        embed_np = embed.detach().cpu().numpy()
        clusterer = SpectralClustering(n_clusters=self.k)
        clusterer.fit(embed_np)

        self.assignments = clusterer.labels_

    def get_membership(self):
        return self.assignments

    def get_loss(self, embed):
        """
        Internal, untuned loss function. This should not be used externally.
        """
        loss = torch.Tensor([0.])

        # L = \sum_i \min_j |x_i - x_j|
        for i, item in enumerate(embed):
            # Build a kd-tree of cluster
            cluster_pts = embed[np.where(self.assignments == self.assignments[i])[0]]
            cluster_pts = cluster_pts.detach().numpy()
            tree = KDTree(cluster_pts)

            distances = [tree.query(x, p=1)[0] for x in cluster_pts]
            loss += sum(distances)

        return loss