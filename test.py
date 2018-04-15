'''
This is an example to detect non-equilibrium regions in the TNF Flame

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from os.path import join
from scipy import interpolate

# path to case, adjust it
case = '/home/max/HDD2_Data/OF4_Simulations/TNF_KIT/SuperMUC/case-06'


def read_data(data_name = 'plane_xy_00.csv'):

    # reads in the data as a Dask dataframe
    data_all_dd = dd.read_csv(join(case, data_name),assume_missing=True)

    global columns
    columns = ['ccx', 'ccy', 'ccz', 'C2H2', 'C2H4', 'C2H6', 'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H',
               'H2',
               'H2O', 'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T', 'f_Bilger', 'Chi', 'PV']

    # consider only the relevant columns
    global data_df
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

    # scale the data!

    global data_sc
    data_sc = data_df - data_df.mean()

    # Different scaling methods: Auto, range, Level, Max, VAST, PARETO
    # see: Parente, Sutherland: Comb. Flame, 160 (2013) 340-350 for more information on scaling
    # we use PARETO scaling here, proven to be the be a good choice: scale data by sqrt of std

    # Pareto
    data_sc = data_sc / np.sqrt(data_sc.std())

    global xArray, yArray, zArray, xi ,yi, zi
    xArray = data_df['ccx']
    yArray = data_df['ccy']
    zArray = data_df['ccz']

    xi = np.linspace(np.min(xArray), np.max(xArray), 5075)
    yi = np.linspace(np.min(yArray), np.max(yArray), 1000)
    zi = np.linspace(np.min(yArray), np.max(yArray), 1000)


read_data('plane_yz_100.csv')

######################################
# plot the field data
######################################
# only the x and y components are relevant


def plot_field(field='T',cmap='jet',plane = 'XY',data_df=data_df):

    if plane == 'XY':

        # for X-Y plane data
        XI, YI = np.meshgrid(xi, yi)

        # these are the regions around the pilot to be masked out
        maskup = (YI > 0.009) & (XI < 0)
        maskdown = (YI < -0.009) & (XI < 0)
        tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
        tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)

        points = np.vstack((xArray,yArray)).T
        field_interp = interpolate.griddata(points,data_df[field],(XI,YI),'linear')
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

    elif plane == 'YZ':
        # for Y-Z plane data
        YI, ZI = np.meshgrid(yi, zi)
        points = np.vstack((yArray,zArray)).T
        field_interp = interpolate.griddata(points,data_df[field],(YI,ZI),'linear')

        plt.figure(figsize=(10,10))

        plt.imshow(field_interp,cmap=cmap)

        plt.title('Field: %s' % field)
        plt.xlabel('x-Axis')
        plt.ylabel('y-Axis')
        plt.colorbar(label=field)

    plt.tight_layout()

    plt.show(block=False)


# function to plot the scatter data
def plot_scatter(field='T',c='k'):
    plt.scatter(data_df['f_Bilger'],data_df[field],s=0.2,c=c,marker='.')
    plt.xlabel('Mixture fraction')
    if field == 'T':
        plt.ylabel('T [K]')
    else:
        plt.ylabel('Mass fraction')
    plt.show(block=False)


# a simple k-Means clustering
from sklearn.cluster import KMeans

def plot_kmeans(n_clusters=10,style='jet', drop = ['T','Chi','PV','ccx','ccy','ccz', 'f_Bilger'], plane='YZ',
                learning_rate = 0.2, scatter_spec = 'f_Bilger'):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    plt.close('all')

    # get the cluster labels
    labels = kmeans.labels_
    #z = data_dd[field].compute()
    zz = labels.reshape(len(y),len(x))

    cmap = plt.get_cmap(style, n_clusters)
    plt.contourf(xx,yy,zz,cmap=cmap)
    plt.title('KMeans: %s clusters' % str(n_clusters))
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar(ticks=range(n_clusters))
    #plt.clim(-0.5, n_clusters- 0.5)
    plt.show(block=False)



def plot_SOM(n_clusters=10,style='jet', drop = ['T','Chi','PV','ccx','ccy','ccz', 'f_Bilger'], plane='YZ',
             learning_rate = 0.2, scatter_spec = 'f_Bilger'):
    from minisom import MiniSom

    global X
    X_ = data_df.copy()
    X_ = X_.drop(drop, axis=1)
    X = X_.values

    global som
    som = MiniSom(n_clusters, 1, X.shape[1], sigma=1.0, learning_rate=0.5)

    # get the cluster labels
    som.train_random(X,200)

    z = []
    for cnt, xy in enumerate(X):
        z.append(som.winner(xy)[0])

    global zz
    zz = np.array(z)

    if plane == 'XY':
        # for X-Y plane data
        plt.figure()
        XI, YI = np.meshgrid(xi, yi)

        # these are the regions around the pilot to be masked out
        maskup = (YI > 0.009) & (XI < 0)
        maskdown = (YI < -0.009) & (XI < 0)
        tipp1 = (XI < 0) & (YI < 0.004) & (YI > 0.00375)
        tipp2 = (XI < 0) & (YI > -0.004) & (YI < -0.00375)

        points = np.vstack((xArray, yArray)).T
        field_interp = interpolate.griddata(points, zz, (XI, YI), 'linear')
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

        YI, ZI = np.meshgrid(yi, zi)
        points = np.vstack((yArray,zArray)).T
        field_interp = interpolate.griddata(points,zz,(YI,ZI),'linear')

        plt.figure(figsize=(10,10))

        cmap = plt.get_cmap(style, n_clusters)
        plt.imshow(field_interp,cmap=cmap)

        plt.title('SOM: %s clusters' % str(n_clusters))
        plt.xlabel('x axis')
        plt.ylabel('y axis')

    plt.colorbar(ticks=range(n_clusters))

    plt.tight_layout()

    plt.figure()
    plt.scatter(data_df['f_Bilger'],data_df[scatter_spec],s=0.5,c=zz,cmap=cmap)

    # plt.figure()
    # plt.title('FGM scatter with SOM cluster')
    # plt.scatter(data_df['f_Bilger'],data_df['T_flamelet'],s=0.5,c=zz,cmap=cmap)

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

