#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:52:01 2019

@author: wesleyz
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd 

path = os.getcwd() # my actual path    
# Read data from file 'filename.csv' 
# (in the same directory that your python process is based)
# Control delimiters, rows, column names with read_csv (see later) 


"features-harem-cd-1.csv"
"labels-harem-cd-1.csv"

filedata = path + '/features-harem-cd-1.csv'
fileLabel = path + '/labels-names-harem-cd-1_.csv'
fullfile = path + '/full-harem-cd-1.csv'

#data = pd.read_csv("features-harem-cd-1.csv") 
data = pd.read_csv(filedata) 
label = pd.read_csv(fileLabel) 
fulldata = pd.read_csv(fullfile) 



def classifica(df):
    # Preview the first 5 lines of the loaded data 
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'label'], df.label, test_size=0.30, random_state=42)
    
    
    
    
    
    
    
    
    maxi = []
    print('B_PESSOA-------------------------------------')
    for j in  ['precision', 'recall', 'f1-score', 'support']: #['precision']:
        bench = []
        fig = plt.figure()    
        for i in range(30,50,1):    
            clf = neighbors.KNeighborsClassifier(i, n_jobs = 20)
            clf.fit(X_train,y_train)
            #clf.fit(df.loc[df['label'] == 2], df.loc[df['label'] == 2].label)
            y_pred = clf.predict(X_test)        
            #y_pred = clf.predict(fulldata.loc[:, fulldata.columns != 'label'])        
            cr = classification_report(y_test, y_pred, output_dict=True)
            #bench.append((i, cr['I_PESSOA'][j], 'j'))        
            bench.append((i, cr['2'][j], 'j'))
            
        benc = pd.DataFrame(bench, columns=['k', 'valor', 'metrica',])
        maxi.append((j, benc.valor.max()))
        #benc.hist()
        ax = plt.axes()
        ax.plot(benc.k, benc.valor, label = j);
        plt.legend()
        #plt.show()
    print(maxi)
    


from sklearn.metrics.pairwise import cosine_similarity



df = fulldata


cosimi = cosine_similarity(df)


for i in range(13, 14):
    zeta = cosimi[i,:]
    zeta = pd.Series(zeta)
    zeta.max()
    zeta.min()
    zeta.hist()




from sklearn import preprocessing


x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
dfS = pd.DataFrame(x_scaled)


#cosimi.sort(axis=0)
#cosimi.sort(axis=1)



#print(cosimi)


#df.loc[df['column_name'] == some_value]
#df = fulldata[['next2W', 'nextW', 'prevW', 'label']]
#df.hist()
#classifica(fulldata)
#from numpy import linalg as LA
#w, v = LA.eigh(df.cov())
#wS = pd.Series(w)
#wS.hist()
#v.shape
#w.shape
#df5 = pd.DataFrame(w)
#df5.plot()


#df5 = pd.DataFrame(v)

import matplotlib.pyplot as plt

#df.cov()

plt.matshow(cosimi)
plt.show()

#plt.matshow(df.corr())
#plt.show()





#benc.hist()


#pivot = bench.pivot_table(index=['k',], columns=['precision', 'recall', 'f1-score'. 'support']) #,fill_value=0) #,  aggfunc=[np.count_nonzero])#,values=["cliente"],aggfunc=[np.count_nonzero], fill_value=0

#pivotVendas = bench.pivot_table(index=['K', ], columns=['precision', 'recall', 'f1-score'. 'support'],fill_value=0,  aggfunc=[sum])#,values=["cliente"],aggfunc=[np.count_nonzero], fill_value=0)

#plt.scatter(x, y )
#plt.legend()
#plt.show()
#clf.fit(data, label)

'''
n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could avoid this ugly
# slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

h = .02  # step size in the mesh

# Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

                            
for weights in ['uniform', 'distance']:
    # we create an instance of Neighbours Classifier and fit the data.
    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X, y)
    clf.fit(data, label)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i, weights = '%s')"
              % (n_neighbors, weights))

plt.show()
'''