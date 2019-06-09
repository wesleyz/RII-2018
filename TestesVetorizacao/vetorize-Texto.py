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
    file = path + '/miniharem/NER-20-portarias_treino.txt'
    
    with open(file, 'r', encoding='utf-8') as infile:
        for line in infile:
            #print(line)
            valor = line.split(" ")#.replace(' ','').decode("ISO-8859-1").split('\n')
            
            
            print(len(valor))
            print(valor)
            if len(valor)>3:                
                cont=0
                print('tamanho split', len(valor))
                documento['word']=valor[0]
                documento['label']=valor[19].replace('\n','')
                documento['tipo']=valor[1].replace('\n','')
                
                
                for tokens in valor:
                    subtokens = tokens.split('=')
                    if len(subtokens) > 1:
                        label=subtokens[0]
                        valor=subtokens[1].replace('\n','')
                        if valor == 'null':
                            valor = 0
                        documento[label]=valor
                        print('Feature: %s ........................... Valor: %s' %(label,valor))
                    else:
                        #print('no label')
                        if cont == 0:
                            print('word: %s' % subtokens[0])
                            cont = cont+1 
                        else:
                            print('Item: %s' % subtokens[0])
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



k, l = LoadArquivsaida()





x_columns = ['word', 'label', 'tipo', 'ini', 'cap', 'simb', 'prevW', 'prevT', 'prevCap', 'nextW', 'nextT', 'nextCap', 'prev2W', 'prev2T', 'prev2Cap', 'next2W', 'next2T', 'next2Cap', 'palpite']
#x_columns = ['word', 'tipo', 'ini', 'cap', 'simb', 'prevW', 'prevT', 'prevCap', 'nextW', 'nextT', 'nextCap', 'prev2W', 'prev2T', 'prev2Cap', 'next2W', 'next2T', 'next2Cap', 'palpite']
df = pd.DataFrame.from_dict(k) #, columns=x_columns)

for col in x_columns:
    print(col)
    df[col] = df[col].astype('category')
    
cat_columns = df.select_dtypes(['category']).columns
dfC = df.copy()
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)


df2 = df.drop(columns=['label'])
df3 = pd.DataFrame(dfC.label)
df2.to_csv('features-NER-20-portarias_treino.csv',  header=True, index=False)
df.label.to_csv('labels-values-NER-20-portarias_treino.csv',  header=True, index=False)
dfC.label.to_csv('labels-names-NER-20-portarias_treino.csv',  header=True, index=False)
df.to_csv('full-NER-20-portarias_treino.csv',  header=True, index=False)





'''
from numpy import linalg as LA
w, v = LA.eigh(df.cov())
wS = pd.Series(w)
wS.hist()
v.shape
w.shape
df5 = pd.DataFrame(w)
df5.plot()
df5 = pd.DataFrame(v)
df5.plot()
df5.hist()
df5.plot()
'''

'''
df = pd.DataFrame.from_dict(k) #, columns=x_columns)
df.to_csv('matriz-features.csv',  header=True)


df2 = df.drop(columns=['label'])
dfLabel = df.label

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
dfV = vectorizer.fit_transform(df2)

from sklearn.model_selection import train_test_split

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing.label import _encode
from sklearn.utils import column_or_1d
x = column_or_1d(df.label, warn=True)
classes_,encoded_values = _encode(x,uniques=np.array(['O', 'B_PESSOA', 'I_PESSOA', 'B_VALOR', 'I_VALOR', 'B_TEMPO',
       'B_LOCAL', 'I_LOCAL', 'B_ORGANIZACAO', 'I_TEMPO', 'I_ORGANIZACAO']),encode=True)
encoded_values, classes_

#(array([0, 1, 2, 1, 0, 1, 2]), ['GA', 'TA', 'SA'])

#comparing with labelencoder, which will sort the labels before encoding
le = LabelEncoder()

le.fit_transform(x),le.classes_

yCategorical = le.fit_transform(x)


X_train, X_test, y_train, y_test = train_test_split(dfV, le.fit_transform(x), test_size=0.33, random_state=42)


from sklearn.neighbors import KNeighborsRegressor
# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=5)
# Fit the model on the training data.
knn.fit(X_train, y_train)
# Make point predictions on the test set using the fit model.
predictions = knn.predict(X_test)


from sklearn.metrics import classification_report
print(classification_report(y_test, predictions))

for i,j in zip(y_test, predictions):
    print(i,j)

'''

