'''
Author: Linhao Meng
Date: 2023-06-12 10:14:22
LastEditTime: 2023-06-13 16:44:52
FilePath: \model\metric.py
Description: Quality metrics used for evaluating DR methods
'''

import numpy as np
from scipy import spatial

# calculate and return distance matrix of X 
def cal_D_matrix(X):
    D_list = spatial.distance.pdist(X, 'euclidean')
    D_matrix = spatial.distance.squareform(D_list)
    return D_matrix

# calculate and return distance matrix of X 
def cal_D_matrix_cosine(X):
    D_list = spatial.distance.pdist(X, 'cosine')
    D_matrix = spatial.distance.squareform(D_list)
    return D_matrix

# calculate distances between x1 and x2 and return a matrix
def cal_D(x1, x2):
    return spatial.distance.cdist(x1, x2, 'euclidean')

# compute (weighted) class centroids
# n is the number of classes, y is the data label numpy array and X is data 2d position
def cal_centroids(y, X):
    y = np.array(y)
    n = np.unique(y).size
    result = []
    for i in range(n):
        indices = np.where(y==i)
        result.append(np.average(X[indices],axis=0))

    return np.array(result)

# calculate projection trustworthiness based on high dimensional distance matrix(dh) and low dimensional distance matrix(dl)
# k defines K nearest neighbors for each point for evaluation
def metric_trustworthiness(k, dh, dl):
    n = dh.shape[0]

    # return indices that could sort the distance matrix
    nn_orig = dh.argsort()
    nn_proj = dl.argsort()

    # extract the indixed of k neighbors 
    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        # return data points are among the K neighbors of point i in 2d space but not among k neighbors in high dimensional space
        U = np.setdiff1d(knn_proj[i], knn_orig[i])

        sum_j = 0
        for j in range(U.shape[0]):
            sum_j += np.where(nn_orig[i] == U[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())

# calculate projection continuity based on high dimensional distance matrix(dh) and low dimensional distance matrix(dl)
# k defines K nearest neighbors for each point for evaluation
def metric_continuity(k, dh, dl):
    n = dh.shape[0]

    nn_orig = dh.argsort()
    nn_proj = dl.argsort()

    knn_orig = nn_orig[:, :k + 1][:, 1:]
    knn_proj = nn_proj[:, :k + 1][:, 1:]

    sum_i = 0

    for i in range(n):
        V = np.setdiff1d(knn_orig[i], knn_proj[i])

        sum_j = 0
        for j in range(V.shape[0]):
            sum_j += np.where(nn_proj[i] == V[j])[0] - k

        sum_i += sum_j

    return float((1 - (2 / (n * k * (2 * n - 3 * k - 1)) * sum_i)).squeeze())

# calculate distance consistency which measures class structure is faithfully conveyed in the 2d space
# y is the label(here we use the predicted class)
# d is the distances between all instances and class centroids or class landmarks
def metric_dsc(y, d):
    n = d.shape[0]
    
    # calculate if data point i is closest to the corresponding class centroid or landmark
    e = 0
    for i in range(n):
        if np.amin(d[i])<d[i][y[i]]:
            e += 1
        
    return float(e/n)
