import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as dd
from sklearn.decomposition import PCA
from data_clustering import data_scaling
from sklearn.cluster import KMeans

df=dd.read_csv('pmCDEFarchives/pmD.scat/D30.Yall',skiprows=3,delim_whitespace=True)
print(df.columns)
columns = ['YC2H2', 'YC2H4', 'YC2H6', 'YCH2CO', 'YCH2O', 'YCH3', 'YCH3OH', 'YCH4',
           'YCO(LIF)', 'YCO2', 'YH', 'YH2', 'YH2O', 'YH2O2', 'YHO2', 'YN2', 'YO', 'YO2', 'YOH', 'T(K)','F']
df=df[list(set(df.columns).intersection(columns))]
df=df.compute()


def npc(df_scaled):
    #df_scaled=data_scaling(df.drop(['F'],axis=1),scaling)
    pca = PCA()
    pca.fit(df_scaled)
    var_ration=pca.explained_variance_ratio_
    for i in range(len(var_ration)):
        if (sum(var_ration[0:i+1])>0.95):
            npc = i+1
            break
    return npc,pca

scaling='Auto'
n_clusters=6
sub=[]
df_scaled=data_scaling(df.drop(['F'],axis=1),scaling)
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(df)
df_scaled['label'] = kmeans.labels_
for i in range(n_clusters):
    data_sub = df_scaled[df_scaled['label'] == i].drop(['label'], axis=1)
    sub.append(npc(data_sub))
sub = np.asarray(sub)

npc,pca=npc(df_scaled)
plt.scatter(df['F'],df['T(K)'],marker='d')
plt.show()

