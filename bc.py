'''
This is an example to detect non-equilibrium regions in the TNF Flame

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd

import os
from scipy import interpolate
from sklearn.cluster import KMeans, DBSCAN
import hdbscan
from sklearn.decomposition import PCA
from data_clustering import data_scaling
from data_clustering import npc


def read_data(data_name, case='./Data'):
    # reads in the data as a Dask dataframe
    data_all_dd = dd.read_csv(os.path.join(case, data_name), assume_missing=True)

    columns = ['ccx', 'ccy', 'ccz', 'C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O',
               'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H', 'H2', 'H2O', 'H2O2', 'HO2',
               'N2', 'O', 'O2', 'OH', 'T', 'f_Bilger', 'Chi', 'PV']

    data_df = data_all_dd[columns].compute()
    data_df = data_df.reset_index(drop=True)

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
def plot_contour(data_name, mesh, data_slt, method='contour', cmap='jet', mask=pd.Series()):
    plane = data_name.split('_')[1]
    if plane == 'xy':

        # for X-Y plane data
        XI, YI = np.meshgrid(mesh.xi, mesh.yi)

        # these are the regions around the pilot to be masked out
        maskup = (YI > 0.009) & (XI < 0)
        maskdown = (YI < -0.009) & (XI < 0)
        tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
        tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)
        points = np.vstack((mesh.xArray, mesh.yArray)).T

        field_interp = interpolate.griddata(points, data_slt, (XI, YI), 'linear')
        field_interp[maskup] = np.nan
        field_interp[maskdown] = np.nan
        field_interp[tipp1] = np.nan
        field_interp[tipp2] = np.nan

        fm = field_interp
        if not mask.empty:
            field_mask = interpolate.griddata(points, mask, (XI, YI), 'linear')
            fm = np.ma.masked_where(field_mask, field_interp)
        plt.figure(figsize=(25, 5))
        plt.imshow(fm, cmap=cmap)

        plt.yticks([0, 462.5, 500, 537.5, 1000], ('0.05', 'D/2', '0', '-D/2', '-0.05'))
        plt.xticks([0, 75, 2 * 75, 6 * 75, 11 * 75, 16 * 75, 21 * 75, 31 * 75],
                   ('-D', '0', 'D', '5D', '10D', '15D', '20D', '30D'))

        plt.title(method + ':' + data_slt.name)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=method)

    elif plane == 'yz':
        # for Y-Z plane data
        YI, ZI = np.meshgrid(mesh.yi, mesh.zi)
        points = np.vstack((mesh.yArray, mesh.zArray)).T

        field_interp = interpolate.griddata(points, data_slt, (YI, ZI), 'linear')

        fm = field_interp
        if not mask.empty:
            field_mask = interpolate.griddata(points, mask, (YI, ZI), 'linear')
            fm = np.ma.masked_where(field_mask, field_interp)
        plt.figure(figsize=(10, 10))
        plt.imshow(fm, cmap=cmap)

        plt.title(method + ':' + data_slt.name)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=method)

    plt.tight_layout()
    plt.show(block=False)


# function to plot the scatter data
def plot_scatter(data_df, y_sc, x_sc='f_Bilger', method=''):
    if x_sc == 'f_Bilger':
        plt.xlabel('Mixture fraction')
    if y_sc == 'T':
        plt.ylabel('T [K]')
    else:
        plt.ylabel('Mass fraction')

    ncs = len(set(data_df['label']).difference([-2]))
    cmp = plt.get_cmap('jet', ncs)
    plt.scatter(data_df[x_sc], data_df[y_sc], s=0.5, c=data_df['label'], cmap=cmp)
    plt.xlim([0, 0.15])
    plt.colorbar(ticks=range(ncs))
    plt.title(method + ' scatter plot ')
    plt.show(block=False)



def plot_cluster(data_name, mesh, data_df, model,
                 method='',
                 mask=pd.Series(),
                 pca=0,
                 drop=['ccx', 'ccy', 'ccz', 'label',
                       'T', 'Chi', 'PV',
                       'f_Bilger', 'non-eq', 'PV_norm', 'Chi_norm', 'PV_compute'
                       ],
                 ):

    X_ = data_df.copy()
    drop = set(X_.columns).intersection(drop)
    X = X_.drop(drop, axis=1)
    X_pca = X

    if pca > 0:
        pca_model = PCA(n_components=pca)
        X_pca = pca_model.fit_transform(X)
        print(pca_model.explained_variance_ratio_.sum())

    # get the cluster labels
    model.fit(X_pca)
    zz = model.labels_
    data_df['label'] = zz

    n_clusters = len(set(zz).difference([-2]))
    cmap = plt.get_cmap('jet', n_clusters)

    plot_contour(data_name, mesh, data_df['label'], cmap=cmap, method=method, mask=mask)

    X['label'] = zz
    sub = pd.DataFrame()
    for i in set(zz):
        data_sub = X[X['label'] == i].drop(['label'], axis=1)
        sub[str(i)] = [npc(data_sub)[0], npc(data_sub)[1], sum(X['label'] == i)]

    plt.show(block=False)

    return sub

def clustering(df,dsc,model):
    drop = ['ccx', 'ccy', 'ccz', 'label',
            'T', 'Chi', 'PV',
            'f_Bilger', 'non-eq', 'PV_norm', 'Chi_norm', 'PV_compute'
            ]
    drop = set(dsc.columns).intersection(drop)

    X = dsc.copy()
    X = X.drop(drop, axis=1)
    dm = X[df['f_Bilger'] > 0.01].copy()


    model.fit(dm)
    dm['label']=model.labels_
    X['label']=dm['label']
    X=X.fillna(-2)

    df['label'] = X['label']
    n_clusters = len(set(df['label']).difference([-2]))
    cmap = plt.get_cmap('jet', n_clusters)


    sub = pd.DataFrame()
    for i in set(model.labels_):
        data_sub = X[X['label'] == i].drop(['label'], axis=1)
        sub[str(i)] = [npc(data_sub)[0], npc(data_sub)[1], sum(X['label'] == i)]

# plot SOM
from minisom import MiniSom

def plot_SOM(data_name, data_df, mesh,style='jet', nc = 5,learning_rate=0.5, sigma =0.5,
             drop=['ccx', 'ccy', 'ccz','T', 'Chi', 'PV','f_Bilger','non-eq','PV_norm','Chi_norm','PV_compute']):

    X_ = data_df.copy()
    X = X_.drop(drop, axis=1)

    model = MiniSom(nc, 1, X.shape[1], sigma=sigma, learning_rate=learning_rate)

    # get the cluster labels
    model.train_random(X, 200)

    z = []
    for cnt, xy in enumerate(X):
        z.append(model.winner(xy)[0])

    # plot the clusters
    cmap = plt.get_cmap('jet', nc)
    plot_field(data_name, mesh, 'SOM', z, cmap)

    plt.figure()
    plt.scatter(data_df['f_Bilger'], data_df['T'], s=0.5, c=zz, cmap=cmap)
    plt.colorbar(ticks=range(n_clusters))
    plt.title('DBSCAN cluster')

    X['label'] = z
    sub=pd.DataFrame()
    for i in set(z):
        data_sub = X[X['label'] == i].drop(['label'], axis=1)
        print(data_sub)
        # sub.append(npc(data_sub))
        sub[str(i)]=npc(data_sub)
    # sub[str(i)] = np.asarray(sub)

    plt.show(block=False)



    return df,cmap,sub


if __name__ == '__main__':
    # data_name = 'plane_xy_00.csv'
    data_name = 'plane_yz_50.csv'

    df, mesh = read_data(data_name)
    dsc = data_scaling(df, 'Auto')
    # dsc = data_scaling(df, 'PARETO')
    # dsc = df.copy()

    # plot_contour(data_name, mesh, df['T'], mask=df['f_Bilger'] < 0.01)

    model = KMeans(n_clusters=5, random_state=42)
    # model = DBSCAN(eps=0.005, min_samples=200)
    # model = hdbscan.HDBSCAN(min_cluster_size=800)
    df,cmap,cluster = clustering(df,dsc,model)

    # plot_contour(data_name, mesh, df['label'], mask=df['f_Bilger'] < 0.01,cmap=cmap)
    # plot_scatter(df, 'T', method='dbscan')


    drop = ['ccx', 'ccy', 'ccz', 'label',
            'T', 'Chi', 'PV',
            'f_Bilger', 'non-eq', 'PV_norm', 'Chi_norm', 'PV_compute'
            ]
    drop = set(dsc.columns).intersection(drop)
    dsc=dsc.drop(drop,axis=1)
    dsc_s=dsc.sample(frac=0.1)
    pca= PCA(n_components=3)
    a=pca.fit_transform(dsc_s)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a[:,0],a[:,1],a[:,2],c=df.loc[dsc_s.index]['label'])
    plt.show()

