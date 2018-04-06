'''
This is for an exemplarily PCA/SOM analysis of combustion data

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from SOM import SOM

# reads in the data as a Dask dataframe
data_all_dd = dd.read_csv('Data/test_data.csv')

x_pos = dd.read_csv('Data/ccx.csv')
y_pos = dd.read_csv('Data/ccy.csv')
z_pos = dd.read_csv('Data/ccz.csv')

columns = ['C2H2', 'C2H4', 'C2H6',  'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H', 'H2', 'H2O',
           'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T']

# consider only the relevant columns
data_dd = data_all_dd[columns]


#####################################
# For a first visualization
#####################################

x=x_pos.compute()
x = x[:50]
y= y_pos.compute()
y=y[::50]

xx, yy = np.meshgrid(x,y)

def plot_field(field):
    plt.close('all')
    # compute as it is a Dask DF
    z = data_dd[field].compute()
    zz = z.values.reshape(len(y),len(x))

    plt.contourf(xx,yy,zz,cmap='jet')
    plt.title('Field: '+field)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar()
    plt.show(block=False)


#####################################
# Centering and Scaling of the data
#####################################

# here the data is centered around its mean
data_centered = data_dd - data_dd.mean()

# Different scaling methods: Auto, range, Level, Max, VAST, PARETO
# see: Parente, Sutherland: Comb. Flame, 160 (2013) 340-350 for more information on scaling
# we use PARETO scaling here, proven to be the be a good choice: scale data by sqrt of std

# Auto
#data_sc = data_centered/data_centered.std()

# Range
#data_sc = data_centered/(data_centered.max() - data_centered.min())

# Max
#data_sc = data_centered/data_centered.max()

# VAST
#data_sc = data_centered/(data_centered.std()/data_centered.mean())

# PARETO
data_sc = data_centered/np.sqrt(data_centered.std())

# finally read the data set and perform previous computations
data_sc = data_sc.compute()

#####################################
# Clustering
#####################################

# create np array from dataframe
X = data_sc.values

# a simple k-Means clustering
from sklearn.cluster import KMeans

def plot_kmeans(n_clusters=10):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    plt.close('all')

    # get the cluster labels
    labels = kmeans.labels_
    #z = data_dd[field].compute()
    zz = labels.reshape(len(y),len(x))

    plt.contourf(xx,yy,zz,cmap='jet')
    plt.title('KMeans: %s clusters' % str(n_clusters))
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar()
    plt.show(block=False)


def plot_SOM(n_clusters=10):
    from minisom import MiniSom
    som = MiniSom(n_clusters, 1, X.shape[1], sigma=1.0, learning_rate=0.5)
    #kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    plt.close('all')

    # get the cluster labels
    som.train_random(X,200)

    z = []
    for cnt, xy in enumerate(X):
        z.append(som.winner(xy)[0])

    zz = np.array(z)
    zz = zz.reshape(len(y),len(x))

    plt.contourf(xx,yy,zz,cmap='jet')
    plt.title('KMeans: %s clusters' % str(n_clusters))
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar()
    plt.show(block=False)


# a clustering based on SOM


# PCA part







