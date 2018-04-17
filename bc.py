'''
This is an example to detect non-equilibrium regions in the TNF Flame

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
#from os.path import join
import os
from scipy import interpolate
from sklearn.cluster import KMeans,DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from data_clustering import data_scaling
from data_clustering import npc


def read_data(data_name, case='./Data'):

    # reads in the data as a Dask dataframe
    data_all_dd = dd.read_csv(os.path.join(case, data_name),assume_missing=True)

    columns = ['ccx', 'ccy', 'ccz', 'C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O',
               'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H', 'H2','H2O', 'H2O2', 'HO2',
               'N2', 'O', 'O2', 'OH', 'T', 'f_Bilger', 'Chi', 'PV']

    data_df = data_all_dd[columns].compute()

    data_df['non-eq'] = 0

    data_df['PV_norm'] = data_df['PV'] / data_df['PV'].max()

    data_df['Chi_norm'] = data_df['Chi'] / data_df['Chi'].max()

    # computed PV
    w_CO = 28.010399999999997
    w_CO2 = 44.009799999999998
    w_H2O = 18.015280000000001
    w_H2 = 2.0158800000000001
    data_df['PV_compute'] = (data_df['CO'] / w_CO + data_df['CO2'] / w_CO2 + data_df['H2O'] / w_H2O) * 1000

    class geo_mesh:
        xArray = data_df['ccx']
        yArray = data_df['ccy']
        zArray = data_df['ccz']

        xi = np.linspace(np.min(xArray), np.max(xArray), 5075)
        yi = np.linspace(np.min(yArray), np.max(yArray), 1000)
        zi = np.linspace(np.min(yArray), np.max(yArray), 1000)
    geo = geo_mesh

    return data_df, geo


######################################
# plot the field data
######################################
def plot_contour(data_name, mesh, data_slt, method='contour', cmap='jet'):

    plane = data_name.split('_')[1]
    if plane == 'xy':

        # for X-Y plane data
        XI, YI = np.meshgrid(mesh.xi, mesh.yi)

        # these are the regions around the pilot to be masked out
        maskup = (YI > 0.009) & (XI < 0)
        maskdown = (YI < -0.009) & (XI < 0)
        tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
        tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)
        points = np.vstack((mesh.xArray,mesh.yArray)).T

        field_interp = interpolate.griddata(points, data_slt, (XI, YI), 'linear')
        field_interp[maskup] = np.nan
        field_interp[maskdown] = np.nan
        field_interp[tipp1] = np.nan
        field_interp[tipp2] = np.nan

        plt.figure(figsize=(25,5))
        plt.imshow(field_interp,cmap=cmap)

        plt.yticks([0,462.5,500,537.5,1000],('0.05','D/2','0','-D/2','-0.05'))
        plt.xticks([0,75,2*75,6*75,11*75,16*75,21*75,31*75], ('-D','0','D','5D','10D','15D','20D','30D'))

        plt.title( method + ':' + data_slt.name)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=method)

    elif plane == 'yz':
        # for Y-Z plane data
        YI, ZI = np.meshgrid(mesh.yi, mesh.zi)
        points = np.vstack((mesh.yArray,mesh.zArray)).T

        field_interp = interpolate.griddata(points, data_slt, (YI, ZI), 'linear')

        plt.figure(figsize=(10,10))
        plt.imshow(field_interp,cmap=cmap)

        plt.title(method + ':' + data_slt.name)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=method)

    plt.tight_layout()
    plt.show(block=False)


# function to plot the scatter data
def plot_scatter(data_df,y_sc, x_sc='f_Bilger',method=''):
    if x_sc =='f_Bilger':
        plt.xlabel('Mixture fraction')
    if y_sc == 'T':
        plt.ylabel('T [K]')
    else:
        plt.ylabel('Mass fraction')

    ncs = len(set(data_df['label']))
    cmp = plt.get_cmap('jet', ncs)
    plt.scatter(data_df[x_sc], data_df[y_sc], s=0.5, c=data_df['label'],cmap=cmp)
    plt.xlim([0, 0.15])
    plt.colorbar(ticks=range(ncs))
    plt.title(method+' scatter plot ')
    plt.show(block=False)


def plot_cluster(data_name, mesh, data_df, model,
                 method='',
                 pca=0,
                 drop=['ccx', 'ccy', 'ccz','label',
                      'T', 'Chi', 'PV',
                      'f_Bilger','non-eq','PV_norm','Chi_norm','PV_compute'
                      ],
                 ):
    X_ = data_df.copy()
    drop = set(X_.columns).intersection(drop)
    X = X_.drop(drop, axis=1)
    X_pca = X

    if pca>0:
        pca_model = PCA(n_components=pca)
        X_pca=pca_model.fit_transform(X)
        print(pca_model.explained_variance_ratio_.sum())

    # get the cluster labels
    model.fit(X_pca)
    zz = model.labels_
    data_df['label'] = zz


    n_clusters=len(set(zz))
    cmap = plt.get_cmap('jet', n_clusters)

    plot_contour(data_name, mesh, data_df['label'], cmap=cmap, method=method)

    X['label'] = zz
    sub=pd.DataFrame()
    for i in set(zz):
        data_sub = X[X['label'] == i].drop(['label'], axis=1)
        sub[str(i)]=[npc(data_sub)[0],npc(data_sub)[1],sum(X['label']==i)]

    plt.show(block=False)

    return sub


if __name__ == '__main__':

    # data_name = 'plane_xy_00.csv'
    data_name = 'plane_yz_50.csv'

    df, mesh = read_data(data_name)
    dsc = data_scaling(df, 'Auto')
    # dsc = data_scaling(df, 'PARETO')
    # dsc = df.copy()

    # plot_field(data_name, mesh, df['CH4'])
    field=df['Chi']
    # plot_contour(data_name, mesh, field)

    dbscan = DBSCAN(eps=0.005, min_samples=200)
    kmeans = KMeans(n_clusters=5, random_state=42)

    sub_d = plot_cluster(data_name, mesh, df, dbscan,method='dbscan')
    plot_scatter(df, 'CO',method='dbscan')
    plt.scatter(df['label'], df['Chi'], s=0.5)
    plt.show()

    # sub_k = plot_cluster(data_name, mesh, dsc, kmeans)
    # df['label']=dsc['label']
    # plot_scatter(df, 'CO',method='kmeans')
    # plt.scatter(df['label'], df['Chi'], s=0.5)
    # plt.show()

