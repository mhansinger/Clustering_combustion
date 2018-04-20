"""
This is for an exemplarily PCA/SOM analysis of combustion data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# from SOM import SOM


def plot_field(field):
    plt.close('all')
    # compute as it is a Dask DF
    z = data_dd[field].compute()
    zz = z.values.reshape(len(y), len(x))

    plt.contourf(xx, yy, zz, cmap='jet')
    plt.title('Field: ' + field)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar()
    plt.show(block=False)


#####################################
# Centering and Scaling of the data
#####################################
def data_scaling(data_dd, case):
    # here the data is centered around its mean
    data_centered = data_dd - data_dd.mean()

    # Different scaling methods: Auto, range, Level, Max, VAST, PARETO
    # see: Parente, Sutherland: Comb. Flame, 160 (2013) 340-350 for more information on scaling
    # we use PARETO scaling here, proven to be the be a good choice: scale data by sqrt of std
    cases = {
        'Auto': 'Auto',
        'Range': 'Range',
        'Max': 'Max',
        'VAST': 'VAST',
        'PARETO': 'PARETO'
    }

    if cases.get(case) == 'Auto':
        data_sc = data_centered / data_centered.std()
    if cases.get(case) == 'Range':
        data_sc = data_centered / (data_centered.max() - data_centered.min())
    if cases.get(case) == 'Max':
        data_sc = data_centered / data_centered.max()
    if cases.get(case) == 'VAST':
        data_sc = data_centered / (data_centered.std() / data_centered.mean())
    if cases.get(case) == 'PARETO':
        data_sc = data_centered / np.sqrt(data_centered.std())

    # finally read the data set and perform previous computations
    # data_sc = data_sc.compute()

    return data_sc


#####################################
# Clustering
#####################################
def plot_cluster(labels, n_clusters=10, style='jet'):
    # z = data_dd[field].compute()
    zz = labels.reshape(len(y), len(x))

    plt.contourf(xx, yy, zz, cmap=discrete_cmap(n_clusters, style))
    plt.title(X[1] + ' scaling, ' + 'KMeans: %s clusters' % str(n_clusters))
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar(ticks=range(n_clusters))
    plt.clim(-0.5, n_clusters - 0.5)
    plt.show(block=False)


# a simple k-Means clustering
def plot_kmeans(n_clusters=10, style='jet'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X[0])
    plt.close('all')

    # get the cluster labels
    labels = kmeans.labels_
    # z = data_dd[field].compute()
    zz = labels.reshape(len(y), len(x))

    plt.contourf(xx, yy, zz, cmap=discrete_cmap(n_clusters, style))
    plt.title(X[1] + ' scaling, ' + 'KMeans: %s clusters' % str(n_clusters))
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar(ticks=range(n_clusters))
    plt.clim(-0.5, n_clusters - 0.5)
    plt.show(block=False)


def plot_SOM(n_clusters=10, style='jet'):
    from minisom import MiniSom
    som = MiniSom(n_clusters, 1, X.shape[1], sigma=1.0, learning_rate=0.5)
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    plt.close('all')

    # get the cluster labels
    som.train_random(X, 200)

    z = []
    for cnt, xy in enumerate(X):
        z.append(som.winner(xy)[0])

    zz = np.array(z)
    zz = zz.reshape(len(y), len(x))

    cmap = plt.get_cmap(style, n_clusters)
    # plt.contourf(xx,yy,zz,cmap=discrete_cmap(n_clusters, style))
    plt.contourf(xx, yy, zz, cmap=cmap)
    plt.title('SOM: %s clusters' % str(n_clusters))
    plt.xlabel('x axis')
    plt.ylabel('y axis')

    plt.colorbar(ticks=range(n_clusters))
    # plt.colorbar(ticks=range(n_clusters))
    # plt.clim(-0.5, n_clusters- 0.5)
    plt.show(block=False)


################################
# helper function for discrete color map
################################
def discrete_cmap(N, base_cmap=None):
    """
    Create an N-bin discrete colormap from the specified input map

    Note that if base_cmap is a string or None, you can simply do
    return plt.cm.get_cmap(base_cmap, N)
    The following works for string, None, or a colormap instance:
    """
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def discrete_matshow(data):
    # get discrete colormap
    cmap = plt.get_cmap('RdBu', np.max(data) - np.min(data) + 1)
    # set limits .5 outside true range
    mat = plt.matshow(data, cmap=cmap, vmin=np.min(data) - .5, vmax=np.max(data) + .5)
    # tell the colorbar to tick at integers
    cax = plt.colorbar(mat, ticks=np.arange(np.min(data), np.max(data) + 1))



def npc(df_scaled,threshold=0.95):
    n_comp=df_scaled.shape[1]
    pca = PCA(n_components=n_comp)
    pca.fit(df_scaled)
    var_ration = pca.explained_variance_ratio_
    cumsum=np.cumsum(var_ration)
    for cnt,i in enumerate(cumsum):
        if(i>threshold):
            return(cnt+1,i)


# # generate data
# a = np.random.randint(1, 9, size=(10, 10))
# discrete_matshow(a)

# PCA part
if __name__ == '__main__':

    # reads in the data as a Dask dataframe
    data_all_dd = dd.read_csv('Data/test_data.csv')

    x_pos = dd.read_csv('Data/ccx.csv')
    y_pos = dd.read_csv('Data/ccy.csv')
    z_pos = dd.read_csv('Data/ccz.csv')

    columns = ['C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H', 'H2', 'H2O',
               'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T']

    # consider only the relevant columns
    data_dd = data_all_dd[columns].compute()

    #####################################
    # For a first visualization
    #####################################

    x = x_pos.compute()
    x = x[:50]
    y = y_pos.compute()
    y = y[::50]

    xx, yy = np.meshgrid(x, y)

    # create np array from dataframe
    # scaling = 'PARETO'
    scaling = 'Auto'
    data_sc = data_scaling(data_dd, scaling)

    # data_sc = data_sc.drop(['T'], axis=1)
    X = (data_sc, scaling)
    tot = npc(data_sc)

    #%%
    n_clusters = 6
    # sub = []
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X[0])
    # X[0]['label'] = kmeans.labels_
    # for i in range(n_clusters):
    #     data_sub = data_sc[data_sc['label'] == i].drop(['label'], axis=1)
    #     sub.append(npc(data_sub))
    # sub = np.asarray(sub)
    # print(sub[:,0].mean())

    plot_kmeans(n_clusters)

    # %%
    from keras_dec import DeepEmbeddingClustering

    c = DeepEmbeddingClustering(n_clusters=6, input_dim=X[0].shape[1])
    #c.initialize(X, finetune_iters=100000, layerwise_pretrain_iters=50000)
    c.initialize(X[0], finetune_iters=10000, layerwise_pretrain_iters=5000)
    # c.cluster(X[0], y=np.random.randint(10,size=X[0].shape[0]))
    c.cluster(X[0])
    labels = c.DEC.predict_classes(X[0])
    plot_cluster(labels,n_clusters=6)
    X[0]['label'] = labels
    sub=[]
    for i in range(n_clusters):
        data_sub = data_sc[data_sc['label'] == i].drop(['label'], axis=1)
        sub.append(npc(data_sub))
    sub = np.asarray(sub)
    print(sub[:,0].mean())
    # %%
