'''
This is an example to detect non-equilibrium regions in the TNF Flame

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from os.path import join
from scipy import interpolate
from sklearn.manifold import TSNE

# a simple k-Means clustering
from sklearn.cluster import KMeans, DBSCAN

# SOM library
from minisom import MiniSom


# path to case, adjust it
path = '/home/max/Python/Clustering_combustion/Data'

# relevant columns
columns = ['ccx', 'ccy', 'ccz', 'C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H',
               'H2',
               'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T', 'f_Bilger', 'Chi']

drop=['T',  'f_Bilger', 'Chi', 'non-eq', 'Chi_norm', 'PV_compute']

class plane_cluster(object):
    def __init__(self,data_name = 'plane_yz_50.csv',path=path, columns = columns):
        # reads in the data as a Dask dataframe
        self.data_name = data_name
        self.data_all_dd = dd.read_csv(join(path, data_name), assume_missing=True)

        self.columns = columns

        self.data_df = self.data_all_dd[self.columns].compute()

        self.data_df['non-eq'] = 0

        #self.data_df['PV_norm'] = data_df['PV'] / data_df['PV'].max()

        self.data_df['Chi_norm'] = self.data_df['Chi'] / self.data_df['Chi'].max()

        # computed PV
        w_CO = 28.010399999999997
        w_CO2 = 44.009799999999998
        w_H2O = 18.015280000000001
        w_H2 = 2.0158800000000001
        self.data_df['PV_compute'] = (self.data_df['CO'] / w_CO + self.data_df['CO2'] / w_CO2 + self.data_df['H2O'] / w_H2O) * 1000

        # Different scaling methods: Auto, range, Level, Max, VAST, PARETO
        # see: Parente, Sutherland: Comb. Flame, 160 (2013) 340-350 for more information on scaling
        # we use PARETO scaling here, proven to be the be a good choice: scale data by sqrt of std

        class geo_mesh:
            xArray = self.data_df['ccx']
            yArray = self.data_df['ccy']
            zArray = self.data_df['ccz']

            xi = np.linspace(np.min(xArray), np.max(xArray), 5075)
            yi = np.linspace(np.min(yArray), np.max(yArray), 1000)
            zi = np.linspace(np.min(yArray), np.max(yArray), 1000)

        self.geo = geo_mesh()


########################################################
    def data_scaling(self, case):
        # here the data is centered around its mean
        data_centered = self.data_df - self.data_df.mean()

        # Different scaling methods: Auto, range, Level, Max, VAST, PARETO
        # see: Parente, Sutherland: Comb. Flame, 160 (2013) 340-350 for more information on scaling
        # we use PARETO scaling here, proven to be the be a good choice: scale data by sqrt of std
        cases = {
            'Auto': 'Auto',
            'Range': 'Range',
            'Max': 'Max',
            'VAST': 'VAST',
            'PARETO': 'PARETO',
            'Off' : 'Off'
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
        if cases.get(case) == 'Off':
            data_sc = self.data_df

        return data_sc


########################################################
    def plot_field(self, field, zz, cmap):
        '''

        :param field: name of the field to plot (str)
        :param zz:    matrix of the cluster or field (nparray)
        :param cmap:  colormap, e.g. 'jet'
        :return:
        '''

        mesh = self.geo

        plane = self.data_name.split('_')[1]
        if plane == 'xy':

            # for X-Y plane data
            XI, YI = np.meshgrid(mesh.xi, mesh.yi)

            # these are the regions around the pilot to be masked out
            maskup = (YI > 0.009) & (XI < 0)
            maskdown = (YI < -0.009) & (XI < 0)
            tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
            tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)

            points = np.vstack((mesh.xArray, mesh.yArray)).T
            # field_interp = interpolate.griddata(points, data_df[field], (XI, YI), 'linear')
            field_interp = interpolate.griddata(points, zz, (XI, YI), 'linear')
            field_interp[maskup] = np.nan
            field_interp[maskdown] = np.nan
            field_interp[tipp1] = np.nan
            field_interp[tipp2] = np.nan

            plt.figure(figsize=(25, 5))

            plt.imshow(field_interp, cmap=cmap)

            plt.yticks([0, 462.5, 500, 537.5, 1000], ('0.05', 'D/2', '0', '-D/2', '-0.05'))
            plt.xticks([0, 75, 2 * 75, 6 * 75, 11 * 75, 16 * 75, 21 * 75, 31 * 75],
                       ('-D', '0', 'D', '5D', '10D', '15D', '20D', '30D'))

            plt.title('Field: %s' % field)
            plt.xlabel('x-Axis')
            plt.ylabel('y-Axis')
            plt.colorbar(label=field)

        elif plane == 'yz':
            # for Y-Z plane data
            YI, ZI = np.meshgrid(mesh.yi, mesh.zi)
            points = np.vstack((mesh.yArray, mesh.zArray)).T
            # field_interp = interpolate.griddata(points, data_df[field], (YI, ZI), 'linear')
            # a= data_df[field]
            field_interp = interpolate.griddata(points, zz, (YI, ZI), 'linear')

            plt.figure(figsize=(10, 10))
            # cmap = plt.get_cmap(style, n_clusters)
            plt.imshow(field_interp, cmap=cmap)

            plt.title('Field: %s' % field)
            plt.xlabel('x-Axis')
            plt.ylabel('y-Axis')
            plt.colorbar(label=field)

        plt.tight_layout()

        plt.show(block=False)

        return cmap

########################################################
    # function to plot the scatter data
    def plot_scatter(self,field='T', c='k'):
        plt.scatter(self.data_df['f_Bilger'], self.data_df[field], s=0.2, c=c, marker='.')
        plt.xlabel('Mixture fraction')
        if field == 'T':
            plt.ylabel('T [K]')
        else:
            plt.ylabel('Mass fraction')
        plt.show(block=False)

########################################################
    # plots for kmeans clustering
    def plot_kmeans(self,n_clusters=10, style='jet', drop=drop, plane='YZ',
                    learning_rate=0.2, scatter_spec='T',scale='Off'):

        X_ = self.data_scaling(scale)
        X_ = X_.drop(drop, axis=1)
        X = X_.values

        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
        plt.close('all')

        # get the cluster labels
        zz = self.kmeans.labels_
        cmap = plt.get_cmap(style, n_clusters)
        self.plot_field('kmeans', zz, cmap)

        plt.figure()
        plt.scatter(self.data_df['f_Bilger'], self.data_df[scatter_spec], s=0.5, c=zz, cmap=cmap)
        plt.title(scatter_spec)
        plt.show(block=False)


########################################################
    # plots for som clustering
    def plot_SOM(self,n_clusters=10, style='jet', drop=drop, plane='YZ',
                 learning_rate=0.2, scatter_spec='T',scale='Off'):

        X_ =  self.data_scaling(scale)
        X_ = X_.drop(drop, axis=1)
        X = X_.values

        self.som = MiniSom(n_clusters, 1, X.shape[1], sigma=1.0, learning_rate=0.5)

        # get the cluster labels
        self.som.train_random(X, 200)

        zz = []
        for cnt, xy in enumerate(X):
            zz.append(self.som.winner(xy)[0])


        cmap = plt.get_cmap(style, n_clusters)
        self.plot_field('SOM', zz, cmap)

        plt.tight_layout()

        plt.figure()
        plt.scatter(self.data_df['f_Bilger'], self.data_df[scatter_spec], s=0.5, c=zz, cmap=cmap)
        plt.title(scatter_spec)
        plt.show(block=False)


########################################################
    # plots for DBSCAN
    def plot_DBSCAN(self, style='jet', drop=drop, plane='YZ',
                 learning_rate=0.2, scatter_spec='T',scale='Off'):

        X_ =  self.data_scaling(scale)
        X_ = X_.drop(drop, axis=1)
        X = X_.values

        self.DBSCAN = DBSCAN(eps=0.3).fit(X)

        zz = self.DBSCAN.labels_

        cmap = plt.get_cmap(style, len(zz))
        self.plot_field('DBSCAN', zz, cmap)

        plt.tight_layout()

        plt.figure()
        plt.scatter(self.data_df['f_Bilger'], self.data_df[scatter_spec], s=0.5, c=zz, cmap=cmap)
        plt.title(scatter_spec)
        plt.show(block=False)


########################################################
    def TSNE(self,n_components = 4, drop=drop):
        X_ = self.data_df.copy()
        X_ = X_.drop(drop, axis=1)
        X = X_.values

        self.X_embedded = TSNE(n_components=n_components).fit_transform(X)








########################################

# #from scalingPV import scalingPV
#
# # helper function
# def find_nearest(array,value):
#     idx = (np.abs(array-value)).argmin()
#     return array[idx]
#
# # read in the T table
# tables = pd.read_csv('tables_pre_801')
# scalingPV = np.loadtxt('scalingParams_pre_801')
#
# tables.convert_objects(convert_numeric=True)
# tables = tables.ix[~(tables['T'] ==801)]
#
# tables_np = tables.values
#
# # this is the T flamelet database
# tables_T = tables_np.reshape(801,801)
#
# # plt.imshow(tables_T,cmap='jet')
# # plt.show(block=False)
#
# norm_vec = np.linspace(0,1,801)
# norm_vec_df = pd.DataFrame(norm_vec,columns=['Entry'])
#
# data_df['T_flamelet'] = 0
#
# T_vec = []
#


# # detection algorithm
# print('Algorithm is running ...')
# for i in range(0,len(data_df)):
#
#     this_f = data_df['f_Bilger'].iloc[i]
#     this_PV = data_df['PV_compute'].iloc[i]
#     this_T = data_df['T'].iloc[i]
#
#     # find nearest value of f and respective index
#     nearest_f = find_nearest(norm_vec,this_f)
#     f_idx = norm_vec_df[norm_vec_df['Entry'] == nearest_f].index[0]
#
#     # find corresponding PVmax
#     this_PVmax = scalingPV[f_idx]
#
#     # normalize PV based on the f value with this_PVmax
#     this_PVnorm = this_PV / this_PVmax
#
#     # get nearest PVNorm
#     nearest_PVnorm = find_nearest(norm_vec,this_PVnorm)
#
#     # get the index of the PV value
#     PV_idx = norm_vec_df[norm_vec_df['Entry'] == nearest_PVnorm].index[0]
#
#     # get the T value from the table and compare it with simulation data
#     this_table_T = tables_T[PV_idx,f_idx]
#
#     # print(this_table_T)
#     # print(data_process['T'].iloc[i])
#     # print(this_f)
#     # print(this_PV)
#     # print('\n')
#
#     #data_process['T_flamelet'].iloc[i] = this_table_T
#     T_vec.append(this_table_T)
#
# print('Done')
#
# data_df['T_flamelet'] = T_vec
#
# non_eq = np.zeros(len(data_df))
#
# # detect the regions where the Temperatures differe!
# for f in range(0,len(data_df)):
#     if (data_df['T_flamelet'].iloc[f]-data_df['T'].iloc[f]) / data_df['T'].iloc[f] > 0.10:
#         non_eq[f] = 1
#
#     elif (data_df['T_flamelet'].iloc[f]-data_df['T'].iloc[f]) / data_df['T'].iloc[f] < -0.10:
#         non_eq[f] = -1
#
#
# data_df['non-eq'] = non_eq








