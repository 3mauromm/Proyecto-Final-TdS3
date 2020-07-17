# -*- coding: utf-8 -*-
"""
Tarea
Tratamiento de Señales 3
"""

#%% Cargación de datos

# Cargar datos
from numpy import genfromtxt
data = genfromtxt('breast-cancer-wisconsin.csv', delimiter=',')
x = data[:,1:10]
y = data[:,10]/2 - 1

# Separar datos
from sklearn.model_selection import train_test_split 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) 

# Normalizar datos 
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()   
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


#%% Extracción de características

# Extracción de características con PCA
from sklearn.decomposition import PCA 
pca = PCA(n_components = 9)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

from numpy import concatenate
x_train_all = concatenate([x_train,x_train_pca],axis=1)
x_test_all = concatenate([x_test,x_test_pca],axis=1)


#%% Selección de características

# Clasificador
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

# Sequential Forward Selection para Características Estimadas
sfs = SFS(kmeans,
           k_features=3, 
           forward=True, 
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=0)
sfs = sfs.fit(x_train, y_train)
sfs_est = sfs.k_feature_idx_

# Sequential Backwards Selection para Características Estimadas
sbs = SFS(kmeans,
           k_features=3, 
           forward=False, 
           floating=False, 
           verbose=0,
           scoring='accuracy',
           cv=0)
sbs = sbs.fit(x_train, y_train)
sbs_est = sbs.k_feature_idx_

# Sequential Forward Selection para Características Extraidas
sfs = sfs.fit(x_train_pca, y_train)
sfs_ext = sfs.k_feature_idx_

# Sequential Backwards Selection para Características Extraidas
sbs = sbs.fit(x_train_pca, y_train)
sbs_ext = sbs.k_feature_idx_

# Sequential Forward Selection para Características Estimadas y Extraidas
sfs = sfs.fit(x_train_all, y_train)
sfs_all = sfs.k_feature_idx_

# Sequential Backwards Selection para Características Estimadas y Extraidas
sbs = sbs.fit(x_train_all, y_train)
sbs_all = sbs.k_feature_idx_


#%% Clasificación de cada caso
from sklearn.metrics import accuracy_score
acc = []

# Características estimadas
kmeans.fit(x_train, y_train)
y_pre = kmeans.predict(x_test)
acc.append(100*accuracy_score(y_test, y_pre))

# Características extraídas
kmeans.fit(x_train_pca, y_train)
y_pre = kmeans.predict(x_test_pca)
acc.append(100*accuracy_score(y_test, y_pre))

# Características estimadas y extraídas
kmeans.fit(x_train_all, y_train)
y_pre = kmeans.predict(x_test_all)
acc.append(100*accuracy_score(y_test, y_pre))

# Características estimadas SFS
kmeans.fit(x_train[:,sfs_est], y_train)
y_pre = kmeans.predict(x_test[:,sfs_est])
acc.append(100*accuracy_score(y_test, y_pre))

# Características estimadas SBS
kmeans.fit(x_train[:,sbs_est], y_train)
y_pre = kmeans.predict(x_test[:,sbs_est])
acc.append(100*accuracy_score(y_test, y_pre))

# Características extraídas SFS
kmeans.fit(x_train_pca[:,sfs_ext], y_train)
y_pre = kmeans.predict(x_test_pca[:,sfs_ext])
acc.append(100*accuracy_score(y_test, y_pre))

# Características extraídas SBS
kmeans.fit(x_train_pca[:,sbs_ext], y_train)
y_pre = kmeans.predict(x_test_pca[:,sbs_ext])
acc.append(100*accuracy_score(y_test, y_pre))

# Características estimadas y extraídas SFS
kmeans.fit(x_train_all[:,sfs_all], y_train)
y_pre = kmeans.predict(x_test_all[:,sfs_all])
acc.append(100*accuracy_score(y_test, y_pre))

# Características estimadas y extraídas SBS
kmeans.fit(x_train_all[:,sbs_all], y_train)
y_pre = kmeans.predict(x_test_all[:,sbs_all])
acc.append(100*accuracy_score(y_test, y_pre))


#%% Despliegue de resultados

from matplotlib.pyplot import bar,ylim, title
bar([1,2,3,4,5,6,7,8,9],acc)
title('K-Means')
ylim(0,100)

#%%

#import matplotlib.pyplot as plt
#from mpl_toolkits import mplot3d
#fig = plt.figure()
#ax = plt.axes(projection='3d')
#
#for i in range(len(y_test)):
#    if y_test[i]==0:
#        ax.scatter3D(x_test[i,sfs_est[0]],x_test[i,sfs_est[1]],x_test[i,sfs_est[2]],color='b')
#    else:
#        ax.scatter3D(x_test[i,sfs_est[0]],x_test[i,sfs_est[1]],x_test[i,sfs_est[2]],color='r')
        





