'''
This is an example to detect non-equilibrium regions in the TNF Flame

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from os.path import join
from scipy import interpolate

# a simple k-Means clustering
from sklearn.cluster import KMeans

# SOM library
from minisom import MiniSom


# path to case, adjust it
path = '/home/max/HDD2_Data/OF4_Simulations/TNF_KIT/SuperMUC/case-06'

# relevant columns
columns = ['ccx', 'ccy', 'ccz', 'C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H',
               'H2',
               'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T', 'f_Bilger', 'Chi']

class plane_cluster(object):
    def __init__(self,data_name = 'plane_yz_50.csv',path=path, columns = columns):
        # reads in the data as a Dask dataframe
        self.data_all_dd = dd.read_csv(join(path, data_name), assume_missing=True)

        self.columns = columns
        self.data_df = self.data_all_dd[columns].compute()

        self.data_df['non-eq'] = 0

        self.data_df['PV_norm'] = data_df['PV'] / data_df['PV'].max()

        self.data_df['Chi_norm'] = data_df['Chi'] / data_df['Chi'].max()

        # computed PV
        w_CO = 28.010399999999997
        w_CO2 = 44.009799999999998
        w_H2O = 18.015280000000001
        w_H2 = 2.0158800000000001
        self.data_df['PV_compute'] = (data_df['CO'] / w_CO + data_df['CO2'] / w_CO2 + data_df['H2O'] / w_H2O) * 1000

        # scale the data!

        self.data_sc = data_df - data_df.mean()

        # Different scaling methods: Auto, range, Level, Max, VAST, PARETO
        # see: Parente, Sutherland: Comb. Flame, 160 (2013) 340-350 for more information on scaling
        # we use PARETO scaling here, proven to be the be a good choice: scale data by sqrt of std

        # Pareto
        self.data_sc = data_sc / np.sqrt(data_sc.std())

        self.xArray = self.data_df['ccx']
        self.yArray = self.data_df['ccy']
        self.zArray = self.data_df['ccz']

        self.xi = np.linspace(np.min(xArray), np.max(xArray), 5075)
        self.yi = np.linspace(np.min(yArray), np.max(yArray), 1000)
        self.zi = np.linspace(np.min(yArray), np.max(yArray), 1000)


    def plot_field(self,field='T', cmap='jet', plane='YZ'):

        if plane == 'XY':

            # for X-Y plane data
            XI, YI = np.meshgrid(self.xi, self.yi)

            # these are the regions around the pilot to be masked out
            maskup = (YI > 0.009) & (XI < 0)
            maskdown = (YI < -0.009) & (XI < 0)
            tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
            tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)

            points = np.vstack((self.xArray, self.yArray)).T
            field_interp = interpolate.griddata(points, self.data_df[field], (XI, YI), 'linear')
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

        elif plane == 'YZ':
            # for Y-Z plane data
            YI, ZI = np.meshgrid(self.yi, self.zi)
            points = np.vstack((self.yArray, self.zArray)).T
            field_interp = interpolate.griddata(points, self.data_df[field], (YI, ZI), 'linear')

            plt.figure(figsize=(10, 10))

            plt.imshow(field_interp, cmap=cmap)

            plt.title('Field: %s' % field)
            plt.xlabel('x-Axis')
            plt.ylabel('y-Axis')
            plt.colorbar(label=field)

        plt.tight_layout()

        plt.show(block=False)

    # function to plot the scatter data
    def plot_scatter(self,field='T', c='k'):
        plt.scatter(self.data_df['f_Bilger'], self.data_df[field], s=0.2, c=c, marker='.')
        plt.xlabel('Mixture fraction')
        if field == 'T':
            plt.ylabel('T [K]')
        else:
            plt.ylabel('Mass fraction')
        plt.show(block=False)

    # # plots for kmeans clustering
    # def plot_kmeans(self,n_clusters=10, style='jet', drop=['T', 'Chi', 'PV', 'ccx', 'ccy', 'ccz', 'f_Bilger'], plane='YZ',
    #                 learning_rate=0.2, scatter_spec='f_Bilger'):
    #     kmeans = self.KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    #     plt.close('all')
    #
    #     # get the cluster labels
    #     labels = self.kmeans.labels_
    #     # z = data_dd[field].compute()
    #     zz = labels.reshape(len(y), len(x))
    #
    #     cmap = plt.get_cmap(style, n_clusters)
    #     plt.contourf(xx, yy, zz, cmap=cmap)
    #     plt.title('KMeans: %s clusters' % str(n_clusters))
    #     plt.xlabel('x axis')
    #     plt.ylabel('y axis')
    #     plt.colorbar(ticks=range(n_clusters))
    #     # plt.clim(-0.5, n_clusters- 0.5)
    #     plt.show(block=False)


    # plots for som clustering
    def plot_SOM(self,n_clusters=10, style='jet', drop=['T', 'Chi', 'PV', 'ccx', 'ccy', 'ccz', 'f_Bilger'], plane='YZ',
                 learning_rate=0.2, scatter_spec='f_Bilger'):

        X_ = data_df.copy()
        X_ = X_.drop(drop, axis=1)
        X = X_.values

        self.som = MiniSom(n_clusters, 1, X.shape[1], sigma=1.0, learning_rate=0.5)

        # get the cluster labels
        som.train_random(X, 200)

        z = []
        for cnt, xy in enumerate(X):
            z.append(som.winner(xy)[0])

        self.zz = np.array(z)

        if plane == 'XY':
            # for X-Y plane data
            plt.figure()
            XI, YI = np.meshgrid(self.xi, self.yi)

            # these are the regions around the pilot to be masked out
            maskup = (YI > 0.009) & (XI < 0)
            maskdown = (YI < -0.009) & (XI < 0)
            tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
            tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)

            points = np.vstack((self.xArray, self.yArray)).T
            field_interp = interpolate.griddata(points, self.zz, (XI, YI), 'linear')
            field_interp[maskup] = np.nan
            field_interp[maskdown] = np.nan
            field_interp[tipp1] = np.nan
            field_interp[tipp2] = np.nan

            plt.figure(figsize=(25, 5))

            cmap = plt.get_cmap(style, n_clusters)
            # plt.contourf(xx,yy,zz,cmap=discrete_cmap(n_clusters, style))
            plt.imshow(field_interp, cmap=cmap)
            plt.title('SOM: %s clusters' % str(n_clusters))
            plt.xlabel('x axis')
            plt.ylabel('y axis')

            plt.yticks([0, 462.5, 500, 537.5, 1000], ('0.05', 'D/2', '0', '-D/2', '-0.05'))
            plt.xticks([0, 75, 2 * 75, 6 * 75, 11 * 75, 16 * 75, 21 * 75, 31 * 75],
                       ('-D', '0', 'D', '5D', '10D', '15D', '20D', '30D'))

        elif plane == 'YZ':

            YI, ZI = np.meshgrid(self.yi, self.zi)
            points = np.vstack((self.yArray, self.zArray)).T
            field_interp = interpolate.griddata(points, self.zz, (YI, ZI), 'linear')

            plt.figure(figsize=(10, 10))

            cmap = plt.get_cmap(style, n_clusters)
            plt.imshow(field_interp, cmap=cmap)

            plt.title('SOM: %s clusters' % str(n_clusters))
            plt.xlabel('x axis')
            plt.ylabel('y axis')

        plt.colorbar(ticks=range(n_clusters))

        plt.tight_layout()

        plt.figure()
        plt.scatter(self.data_df['f_Bilger'], self.data_df[scatter_spec], s=0.5, c=zz, cmap=cmap)

        plt.show(block=False)



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








