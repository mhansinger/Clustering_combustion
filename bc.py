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
# path to case, adjust it
case = './Data'


def read_data(data_name = 'plane_xy_00.csv'):

    # reads in the data as a Dask dataframe
    data_all_dd = dd.read_csv(os.path.join(case, data_name),assume_missing=True)

    #global columns
    columns = ['ccx', 'ccy', 'ccz', 'C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H',
               'H2',
               'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T', 'f_Bilger', 'Chi', 'PV']

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
    return data_df,geo


######################################
# plot the field data
######################################
# only the x and y components are relevant
def plot_field(data_name, mesh, field, zz,cmap):

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
        # field_interp = interpolate.griddata(points, data_df[field], (XI, YI), 'linear')
        field_interp = interpolate.griddata(points,zz, (XI, YI), 'linear')
        field_interp[maskup] = np.nan
        field_interp[maskdown] = np.nan
        field_interp[tipp1] = np.nan
        field_interp[tipp2] = np.nan

        plt.figure(figsize=(25,5))

        plt.imshow(field_interp,cmap=cmap)

        plt.yticks([0,462.5,500,537.5,1000],('0.05','D/2','0','-D/2','-0.05'))
        plt.xticks([0,75,2*75,6*75,11*75,16*75,21*75,31*75], ('-D','0','D','5D','10D','15D','20D','30D'))

        plt.title('Field: %s'%field)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=field)

    elif plane == 'yz':
        # for Y-Z plane data
        YI, ZI = np.meshgrid(mesh.yi, mesh.zi)
        points = np.vstack((mesh.yArray,mesh.zArray)).T
        #field_interp = interpolate.griddata(points, data_df[field], (YI, ZI), 'linear')
        # a= data_df[field]
        field_interp = interpolate.griddata(points, zz, (YI, ZI), 'linear')

        plt.figure(figsize=(10,10))
        # cmap = plt.get_cmap(style, n_clusters)
        plt.imshow(field_interp,cmap=cmap)

        plt.title('Field: %s' % field)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=field)
        # plt.colorbar(label=zz.name)

    plt.tight_layout()

    plt.show(block=False)

    return cmap


# function to plot the scatter data
def plot_scatter(data_df, field='T',c='k'):
    plt.scatter(data_df['f_Bilger'],data_df[field],s=0.2,c=c,marker='.')
    plt.xlabel('Mixture fraction')
    if field == 'T':
        plt.ylabel('T [K]')
    else:
        plt.ylabel('Mass fraction')
    plt.show(block=False)


def plot_kmeans(data_name, df, data_df, mesh, nc=10,
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

    model = KMeans(n_clusters=nc, random_state=42).fit(X_pca)

    # get the cluster labels
    zz = model.labels_
    cmap = plt.get_cmap('jet', nc)
    plot_field(data_name, mesh, 'kmeans', zz, cmap)

    plt.figure()
    # plt.scatter(data_df['f_Bilger'], data_df['T'], s=0.5, c=zz, cmap=cmap)
    plt.scatter(df['f_Bilger'], df['CO'], s=0.5, c=zz, cmap=cmap)
    plt.xlim([0, 0.15])
    plt.colorbar(ticks=range(nc))
    plt.title('kmeans cluster (pca:'+str(pca)+')')

    X['label'] = zz
    data_df['label'] = zz
    sub=pd.DataFrame()
    for i in set(zz):
        data_sub = X[X['label'] == i].drop(['label'], axis=1)
        sub[str(i)]=[npc(data_sub)[0],npc(data_sub)[1],sum(X['label']==i)]

    plt.show(block=False)
    return sub


def plot_DBSCAN(data_name, df,data_df, mesh,
                pca=0,
                drop=['ccx', 'ccy', 'ccz','label',
                      'T', 'Chi', 'PV',
                      'f_Bilger','non-eq','PV_norm','Chi_norm','PV_compute'
                      ],
                EPS=0.2, MINS=200):

    X_ = data_df.copy()
    drop = set(X_.columns).intersection(drop)
    X = X_.drop(drop, axis=1)
    X_pca = X

    if pca>0:
        pca_model = PCA(n_components=pca)
        X_pca=pca_model.fit_transform(X)
        print(pca_model.explained_variance_ratio_.sum())

    # model = DBSCAN(eps=EPS, min_samples=MINS).fit(X_pca)
    model = hdbscan.HDBSCAN(min_samples=MINS,gen_min_span_tree=True).fit(X_pca)

    # get the cluster labels
    zz = model.labels_
    n_clusters=len(set(zz))
    cmap = plt.get_cmap('jet', n_clusters)
    plot_field(data_name, mesh, 'dbsan', zz, cmap)

    plt.figure()
    # plt.scatter(data_df['f_Bilger'], data_df['T'], s=0.5, c=zz, cmap=cmap)
    plt.scatter(df['f_Bilger'], df['CO'], s=0.5, c=zz, cmap=cmap)
    plt.xlim([0, 0.15])
    plt.colorbar(ticks=range(n_clusters))
    plt.title('DBSCAN cluster (pca:'+str(pca)+')')
    plt.show()

    X['label'] = zz
    data_df['label'] = zz
    sub=pd.DataFrame()
    for i in set(zz):
        data_sub = X[X['label'] == i].drop(['label'], axis=1)
        sub[str(i)]=[npc(data_sub)[0],npc(data_sub)[1],sum(X['label']==i)]

    # plt.figure()
    # plt.scatter(df[X['label'] == 1]['f_Bilger'],df[X['label'] == 1]['T'],s=0.5,cmap='jet')

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
    cmap='jet'
    field=df['Chi']
    #plot_field(data_name, mesh,field.name, field,cmap)

    # pca=PCA(n_components=5)
    # X=pca.fit_transform(df)
    sub_d = plot_DBSCAN(data_name, df, dsc, mesh, EPS=0.5, MINS=400)

    # plt.scatter(dsc['label'], df['Chi'], s=0.5)
    # plt.show()
    # # sub_d = plot_DBSCAN(data_name, df, dsc, mesh, pca=6, EPS=0.5, MINS=400)
    # sub_k = plot_kmeans(data_name, df, dsc, mesh, nc=5)
    # # sub_k = plot_kmeans(data_name, df, dsc, mesh, pca=7, nc=5,)
    # plt.scatter(dsc['label'], df['Chi'], s=0.5)
    # plt.show()