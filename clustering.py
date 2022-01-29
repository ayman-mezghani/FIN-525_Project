import os
from os.path import exists

import community
import dask
import networkx as nx
import numpy as np
import pandas as pd
from numpy import linalg as LA
from sklearn import metrics
from tqdm import tqdm

dask.config.set(scheduler='processes')


# Idea is to remove low eigenvalues that are considered as noise
def compute_C_minus_C0(lambdas, v, lambda_plus, removeMarketMode=True):
    N = len(lambdas)
    C_clean = np.zeros((N, N))

    order = np.argsort(lambdas)
    lambdas, v = lambdas[order], v[:, order]

    v_m = np.matrix(v)

    # note that the eivenvalues are sorted
    for i in range(1 * removeMarketMode, N):
        if lambdas[i] > lambda_plus:
            C_clean = C_clean + lambdas[i] * np.dot(v_m[:, i], v_m[:, i].T)
    return C_clean


def LouvainCorrelationClustering(R):  # R is a matrix of return
    N = R.shape[1]
    T = R.shape[0]

    q = N * 1. / T
    lambda_plus = (1. + np.sqrt(q)) ** 2

    C = R.corr()
    lambdas, v = LA.eigh(C)

    C_s = compute_C_minus_C0(lambdas, v, lambda_plus)

    mygraph = nx.from_numpy_matrix(np.abs(C_s))
    partition = community.community_louvain.best_partition(mygraph, random_state=29)

    DF = pd.DataFrame.from_dict(partition, orient="index", columns=['Cluster'])
    DF.index = R.columns
    return (DF)


# Check if we already have the clustering or not and compute louvain cluster
@dask.delayed
def get_Louvain_cluster(R, filename, t_0, t_1, day_state=False):
    # t_0 and t_1 = window
    # day_state = TRUE -> market state clustering

    # Check if cluster already exists
    if (exists(filename) and os.path.getsize(filename) > 0):

        # Import clustering
        DF = pd.read_parquet(filename)

    else:

        # Do clustering method and saves it
        rolling_data = R.iloc[(R.index >= t_0) & (R.index <= t_1)]

        if day_state:
            rolling_data = rolling_data.T

        DF = LouvainCorrelationClustering(rolling_data)
        DF.to_parquet(filename)

    return (DF)


@dask.delayed
def RolledCluster(R, cluster_period, filepath, day_state=False, keep_all_info=False, lag_max=20):
    # R = dataframe
    # cluster_period = length of rolling window
    # day_state = TRUE -> market state clustering
    # keep_all_info = TRUE -> rolling window expanding instead of just rolling (pas sur detre utile)
    # lag_max = number of clustering

    # Create list of clusters
    liste = []

    #
    if keep_all_info:
        if day_state:
            for lag in tqdm(range(1, lag_max)):
                # initilisation of window
                t_0 = R.index[0]
                t_1 = R.index[0 + lag + cluster_period]

                # compute convenient filename
                filename = filepath + "/{}_{}_cluster_day_keep_all_info.parquet".format(t_0, t_1)

                # compute cluster
                DF = get_Louvain_cluster(R, filename, t_0, t_1, day_state)

                # add cluster to list
                liste.append(DF)
        else:
            for lag in tqdm(range(1, lag_max)):
                t_0 = R.index[0]
                t_1 = R.index[0 + lag + cluster_period]

                filename = filepath + "/{}_{}_cluster_keep_all_info.parquet".format(t_0, t_1)

                DF = get_Louvain_cluster(R, filename, t_0, t_1, day_state)
                liste.append(DF)
    else:
        if day_state:
            for lag in tqdm(range(1, lag_max)):
                t_0 = R.index[0 + lag]
                t_1 = R.index[0 + lag + cluster_period]

                filename = filepath + "/{}_{}_cluster_day.parquet".format(t_0, t_1)

                DF = get_Louvain_cluster(R, filename, t_0, t_1, day_state)
                liste.append(DF)
        else:
            for lag in tqdm(range(1, lag_max)):
                t_0 = R.index[0 + lag]
                t_1 = R.index[0 + lag + cluster_period]

                filename = filepath + "/{}_{}_cluster.parquet".format(t_0, t_1)

                DF = get_Louvain_cluster(R, filename, t_0, t_1, day_state)
                liste.append(DF)
    return (liste)


# Renvoie une liste de clusters
# Choose the assets (thus N ), choose the calibration length
# (T = N/3 is a good choice for market states, T = 3N is a good choice for asset classification)


# Compute the rolling ARI of the clusters
def RolledARI(liste):
    # Create list of ARI
    ARI = []

    for element in tqdm(range(0, len(liste) - 1)):
        # Compare consecutive clustering and add measure
        new_df = pd.merge(liste[element], liste[element + 1], left_index=True, right_index=True)
        ARI.append(metrics.adjusted_rand_score(new_df["Cluster_x"], new_df["Cluster_y"]))

    return (ARI)


# fonction qu'on avait fait mais pas sur que ca serve dans notre cas
def compute_R(events, tau_max=1000, dtau=1):
    taus = range(1, tau_max, dtau)

    R = [-(events["mid"].diff(-tau) * events["s"]).mean() for tau in taus]

    return np.array(R)


def compute_R_h(events):
    R = pd.pivot_table(events.groupby(events.index.hour).apply(compute_R).apply(pd.Series),
                       columns=events.groupby(events.index.hour).apply(compute_R).apply(pd.Series).index)
    return (R)


# pour linstant renvoie simplement une matrice de similarite
# l'entree (i,j) correspond au pourcentage d'éléments du cluster 2j qui appartiennent aussi au cluster 1i
def keep_cluster_number(cluster1, cluster2):
    # rename clusters based on similarity

    # compute number of clusters within each clustering period
    n = cluster1["Cluster"].unique().max()
    m = cluster2["Cluster"].unique().max()

    # dummy constant to avoid changing twice clusters later
    cst = 10000

    # Create matrix of similarity
    matrix = np.zeros((n + 1, m + 1))

    for i in range(n + 1):
        for j in range(m + 1):
            merged = pd.merge(cluster1, cluster2, left_index=True, right_index=True)

            # get number of days/ticker belonging to cluster x_i and y_j
            num = len(merged[(merged["Cluster_y"] == j) & (merged["Cluster_x"] == i)])

            # divide by number of elements in cluster y_j
            denom = len(merged[(merged["Cluster_y"] == j)])

            matrix[i][j] = num / denom
            # entry (i,j) is the % of elements in y_j coming from x_i

    # get a copy of cluster2 that we change
    new_cluster2 = cluster2.copy(deep=True)

    # inititialize lists of cluster values
    used_values = []
    unrenamed_cluster = []

    # array with number of element in each cluster
    tmp = []
    for i in range(0, m + 1):
        tmp.append(len(cluster2[cluster2["Cluster"] == i]))

    # transform into df
    dff = pd.DataFrame(data=tmp)

    # sort values by ascending number of element
    # relabel cluster based on most important (size) cluster

    dff = dff.sort_values(by=0, ascending=False)

    for i in dff.index:

        # get the position of the highest similarity
        value = np.argmax(matrix[:, i])

        # verify that this value is not already attributed to another cluster
        if not np.isin(value, used_values):

            value_cluster = value

            # update the used values
            used_values.append(value)

            # change the new clusters
            # add dummy constant to avoid changing twice clusters
            new_cluster2[new_cluster2["Cluster"] == i] = (value_cluster + cst)

        else:
            # update cluster that need to be allocated a number
            unrenamed_cluster.append(i)

    # all possible values for cluster number
    possible_values = range(0, m + 1)

    for element in unrenamed_cluster:
        # get smallest unused value
        value_cluster = np.min(np.where(~np.isin(possible_values, used_values)))

        # update used values
        used_values.append(value)

        # change cluster number
        new_cluster2[new_cluster2["Cluster"] == element] = (value_cluster + cst)

    # substract cst to get actual clusters numbers
    new_cluster2 = new_cluster2 - cst

    return (new_cluster2)


# One time prediction of cluster belonging
# lines = period t
# col = period t+1

def one_step_pred(cluster):
    # get number of clusters
    m = cluster["Cluster"].unique().max()

    # create square matrix
    matrix = np.zeros((m + 1, m + 1))

    for p in range(len(cluster) - 1):
        i = cluster["Cluster"].iloc[p]
        tmp = p + 1
        j = cluster["Cluster"].iloc[tmp]

        # adds 1 when you go from state i to state j
        matrix[i][j] = matrix[i][j] + 1

    # divide by total number of changes
    matrix = matrix / (len(cluster) - 1)

    # matrix is now a probability matrix
    # matrix(i,j) is the probability to go from state i to state j

    return (matrix)
