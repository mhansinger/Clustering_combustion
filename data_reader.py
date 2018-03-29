# reads in the data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd

data_all_dd = dd.read_csv('Data/test_data.csv')

x_pos = dd.read_csv('Data/ccx.csv')
y_pos = dd.read_csv('Data/ccy.csv')
z_pos = dd.read_csv('Data/ccz.csv')

columns = ['C2H2', 'C2H4', 'C2H6',  'CH2CO', 'CH2O', 'CH3', 'CH3OH', 'CH4', 'CO', 'CO2', 'H', 'H2', 'H2O',
           'H2O2', 'HO2', 'N2', 'O', 'O2', 'OH', 'T']

data_dd = data_all_dd[columns]

# plotting

x=x_pos.compute()
x = x[:50]
y= y_pos.compute()
y=y[::50]

xx, yy = np.meshgrid(x,y)

def plot_field(field):
    plt.close('all')
    z = data_dd[field].compute()
    zz = z.values.reshape(len(y),len(x))

    plt.contourf(xx,yy,zz,cmap='jet')
    plt.title('Field: '+field)
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    plt.colorbar()
    plt.show(block=False)



#SOM part







