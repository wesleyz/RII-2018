#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 19:39:04 2019

@author: wesleyz
"""

import os
path = os.getcwd() # my actual path    
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer    
from sklearn.model_selection import train_test_split
import scipy.io as sio
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def LoadArquivsaida(caminho):    
    colecao  = []    
    documento = {}
    #file = path + '/Senhora/TreinoPVetorizar.txt'    
    #file = path + '/HAREM/TreinoPVetorizar.txt'    
    file = path + caminho    
    
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
    infile.close()
    return colecao, documento

def SaveData(X,y_train ):    
    sio.mmwrite('matrix.mtx', X)
    y_train.to_csv('label.csv', sep=';', encoding='utf-8', index=False)


def main(caminho):
    lstDicWords, l = LoadArquivsaida(caminho)
    df = pd.DataFrame.from_dict(lstDicWords)
    df.to_csv('matrizBruta.csv',sep=';', encoding='utf-8', index=False)
    y_train = df.label
    x_columns = ['cap', 'ini', 'label', 'next2Cap', 'next2T', 'next2W', 'nextCap', 'nextT', 'nextW', 'palpite', 'prev2Cap', 'prev2T', 'prev2W', 'prevCap', 'prevT', 'prevW', 'simb',  'tipo', 'word'] 
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
        
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(corpus2)
    
    y = y_train
    
    #print(X)
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
    
    
    
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression()
    
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    
    y_pred = logreg.predict(X_test)
    #print(metrics.accuracy_score(y_test, y_pred))
    
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    #print(metrics.accuracy_score(y_test, y_pred))
    
    
    # try K=1 through K=25 and record testing accuracy
    k_range = range(3, 9)
    
    # We can create Python dictionary using [] or dict()
    scores = []
    
    # We use a loop through the range 1 to 26
    # We append the scores in the dictionary
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='euclidean')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))
    
    #print(scores)
    
    # import Matplotlib (scientific plotting library)
    
    
    # allow plots to appear within the notebook
    #%matplotlib inline
    
    # plot the relationship between K and testing accuracy
    # plt.plot(x_axis, y_axis)
    plt.plot(k_range, scores)
    
    plt.legend((documentos[0],documentos[1],documentos[2], documentos[3],   ), )    
    plt.legend((documentos[0],documentos[1],  ), )    
        
    
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')
    print('caminho: %s\n'% caminho)
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    


#file = path + '/Senhora/TreinoPVetorizar.txt'    
#file = path + '/1506052006mul.txt'    
#documentos = ['/Senhora/TreinoPVetorizar.txt', '/caixa-saida/ArqsTeste/SampleSenhora/SampleSenhora_00_features.txt']
documentos = ['/1506052006mul.txt','/Senhora/TreinoPVetorizar.txt'] #,  '/HAREM/TreinoPVetorizar.txt']
documentos = ['/ArqsTeste/portaria1/portaria1.txt', '/ArqsTeste/portaria2/portaria2.txt', '/ArqsTeste/portaria3/portaria3.txt', '/ArqsTeste/portaria4/portaria4.txt'] #, '/ArqsTeste/portaria4/portaria4.txt', '/ArqsTeste/1609102006con/1609102006con.txt', '/ArqsTeste/1605082006mul/1605082006mul.txt', '/ArqsTeste/1525092006con/1525092006con.txt', '/ArqsTeste/1506052006mul/1506052006mul.txt', '/ArqsTeste/1505082006mul/1505082006mul.txt']
documentos = ['/ArqsTeste/portaria1/portaria1.txt'] #, '/ArqsTeste/portaria2/portaria2.txt'] #, '/ArqsTeste/portaria3/portaria3.txt', '/ArqsTeste/portaria4/portaria4.txt'] #, '/ArqsTeste/portaria4/portaria4.txt', '/ArqsTeste/1609102006con/1609102006con.txt', '/ArqsTeste/1605082006mul/1605082006mul.txt', '/ArqsTeste/1525092006con/1525092006con.txt', '/ArqsTeste/1506052006mul/1506052006mul.txt', '/ArqsTeste/1505082006mul/1505082006mul.txt']
documentos = ['//1506052006mul.txt']
#documentos = ['/ArqsTeste/1609102006con/1609102006con.txt', '/ArqsTeste/1605082006mul/1605082006mul.txt', '/ArqsTeste/1525092006con/1525092006con.txt', '/ArqsTeste/1506052006mul/1506052006mul.txt', '/ArqsTeste/1505082006mul/1505082006mul.txt']
for i in documentos:
    main(i)
    
    
import seaborn as sns; sns.set()  

