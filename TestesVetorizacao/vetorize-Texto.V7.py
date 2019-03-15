#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:39:04 2019

@author: wesleyz
"""

import os
path = os.getcwd() # my actual path    
import pandas as pd


def LoadArquivsaida():    
    colecao  = []
    
    documento = {}
    file = path + '/Senhora/TreinoPVetorizar.txt'    
    #file = path +'/ArqsTeste/portaria1/portaria1.txt'
    #file = path + '/ArqsTeste/1506052006mul/1506052006mul.txt'
    #file = path +'/ArqsTeste/1606112006con/1606112006con.txt'
    #file = path +'/ArqsTeste/1605082006mul/1605082006mul.txt'
    #file = path + '/HAREM/TreinoPVetorizar.txt'

    
    with open(file, 'r', encoding='utf-8') as infile:
        for line in infile:
            #print(line)
            valor = line.split(" ")#.replace(' ','').decode("ISO-8859-1").split('\n')
            if len(valor)>1:                
                cont=0
                #print('tamanho split', len(valor))
                documento['word']=valor[0]
                auxLb = valor[17].replace('\n','').split('=')
                documento['label']= auxLb[1]
                documento['tipo']=valor[1].replace('\n','')
                
                
                for tokens in valor:
                    subtokens = tokens.split('=')
                    if len(subtokens) > 1:
                        label=subtokens[0]
                        valor=subtokens[1].replace('\n','')
                        if valor == 'null':
                            valor = 0
                        documento[label]=valor
                        #print('Feature: %s ........................... Valor: %s' %(label,valor))
                    else:
                        #print('no label')
                        if cont == 0:
                            #print('word: %s' % subtokens[0])
                            cont = cont+1 
                        else:
                            #print('Item: %s' % subtokens[0])
                            cont = cont+1 
                aux = documento.copy()
                colecao.append(aux)
                documento.clear()
                
                #documento{label} = subtokens[1]
                
                
            '''
            valor = line.replace('\n','').split(";")
            nomeArquivo = valor[0].strip()
            documento['fileName'] = nomeArquivo             
            pageLixo = valor[1]          
            documento['ano'] = valor[2]            
            tituloArquivo = valor[3].replace(';', '').strip() 
            tituloArquivo = re.sub('\W+',' ', tituloArquivo )
            documento['titulo'] = tituloArquivo
            documento['label'] = valor[4].strip()             
            documento['pA'] = valor[5]
            documento['pB'] = valor[6]            
            cont=cont+1
            pageClean = [int(s) for s in pageLixo.split() if s.isdigit()]
            if len(pageClean) > 0:
                paginaArquivo = pageClean[0]  #1 pagina, titulo, 
                documento['paginas'] = paginaArquivo               
                aux = documento.copy()
                colecao.append(aux)
                documento.clear()
            '''
    infile.close()
    return colecao, documento


lstDicWords, l = LoadArquivsaida()

df = pd.DataFrame.from_dict(lstDicWords)

y_train = df.label


x_columns = ['cap', 'ini', 'label', 'next2Cap', 'next2T', 'next2W', 'nextCap', 'nextT', 'nextW', 'palpite', 'prev2Cap', 'prev2T', 'prev2W', 'prevCap', 'prevT', 'prevW', 'simb',  'tipo', 'word'] #['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']

corpus = []
aux = []
corpus2 = []

for feat in lstDicWords:
    for label in x_columns:
        aux.append(feat[label])
    corpus.append(aux.copy())
    aux.clear()
        
        
for itens in corpus:
    str1 = ' '.join(str(e) for e in itens)
    corpus2.append(str1)
    str1=''
    
from sklearn.feature_extraction.text import CountVectorizer    
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus2)

y = y_train
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)

#from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier




# try K=1 through K=25 and record testing accuracy
k_range = range(2,8)

# We can create Python dictionary using [] or dict()
scores = []
best =  []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    auxScore = metrics.accuracy_score(y_test, y_pred)
    scores.append(auxScore)
    best.append((k, auxScore))
    

print(scores)

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt

# allow plots to appear within the notebook
#%matplotlib inline

# plot the relationship between K and testing accuracy
# plt.plot(x_axis, y_axis)

plt.plot(k_range, scores)



plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')



from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))


best = dict(best)
print('Best Value for K is: %d' % max(best, key=best.get))
print('Acc Score: %f' % max(scores))



'''
import scipy.io as sio
sio.mmwrite('matrix.mtx', X)

y_train.to_csv('label.csv', sep=';', encoding='utf-8', index=False)
'''

from sklearn.metrics.pairwise import cosine_similarity
simil = cosine_similarity(X)



import seaborn as sns; sns.set()


linha = df.index[df['label'] == 'I_PESSOA'].tolist()
coluna = simil.shape[1]
reduz = []
for i in linha:
    for j in range(0, coluna):
        if i != j:
        #if True:
            valorSm = simil[i][j]
            #if True:
            if (valorSm >= 0.80) or (valorSm <= 0.20):
                reduz.append((i,j,valorSm))
            
            

auxColuna = []    
auxItem = []
for rd in reduz:
    #print(rd[0])
    #k = rd[1]
    auxColuna.append(rd[2])
    indice = rd[1]
    if indice not in auxItem:
        auxItem.append(indice)
    #auxColuna.append(rd[1])
    
j = pd.Series(auxColuna)
kItem = pd.Series(auxItem)

#ax = j.hist()  # s is an instance of Series
#fig = ax.get_figure()
#nome = 'graf.'+'pdf'
#fig.savefig(nome)





dfReduz = df.ix[auxItem]
#dfReduz["label"] = dfReduz["label"].astype('category')


y_trainReduz = dfReduz.label




from sklearn.base import TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction import DictVectorizer


class RowIterator(TransformerMixin):
    """ Prepare dataframe for DictVectorizer """
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return (row for _, row in X.iterrows())


vectorizer = make_pipeline(RowIterator(), DictVectorizer())

# now you can use vectorizer as you might expect, e.g.




vectorizerReduz = CountVectorizer()
X = vectorizer.fit_transform(dfReduz)

y = y_trainReduz
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)


k_range = range(2,8)

# We can create Python dictionary using [] or dict()
scores = []
best =  []

# We use a loop through the range 1 to 26
# We append the scores in the dictionary
for k in k_range:
    
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    auxScore = metrics.accuracy_score(y_test, y_pred)
    scores.append(auxScore)
    best.append((k, auxScore))

print(classification_report(y_test, y_pred))


plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')

best = dict(best)
print('Best Value for K is: %d' % max(best, key=best.get))
print('Acc Score: %f' % max(scores))
#ax = sns.heatmap(simil,  vmin=0, vmax=1, cmap="YlGnBu")


df["label"] = df["label"].astype('category')

import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt


reg = linear_model.LinearRegression()
#reg.fit(X,y)
#sns.heatmap(simil, cmap="YlGnBu")

