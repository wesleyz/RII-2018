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
    file = path + '/HAREM/TreinoPVetorizar.txt'
    
    with open(file, 'r', encoding='utf-8') as infile:
        for line in infile:
            #print(line)
            valor = line.split(" ")#.replace(' ','').decode("ISO-8859-1").split('\n')
            if len(valor)>1:                
                cont=0
                print('tamanho split', len(valor))
                documento['word']=valor[0]
                documento['label']=valor[18].replace('\n','')
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
    
from sklearn.feature_extraction.text import CountVectorizer    
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus2)


#print(vectorizer.get_feature_names())
#print(X.toarray())     

dfX = pd.DataFrame.from_dict(X.toarray())

from sklearn.neighbors import KNeighborsClassifier
for i in range(2,10):
    neigh = KNeighborsClassifier(n_neighbors=i, n_jobs=20)
    print('n of knn: %d' % i)
    neigh.fit(dfX,y_train)
    print('Score of knn %f' % neigh.score(dfX,y_train))


    

df.loc[df['label'] == 'I_PESSOA', 'label'] = 1
df.loc[df['palpite'] == 'I_PESSOA', 'palpite'] = 1
df.loc[df['palpite'] == 'O', 'palpite'] = 0
df.loc[df['label'] == 'O', 'label'] = 0
'''    



df.to_csv('matriz-features.csv',  header=True)

#y_train, y_test = data_train.target, data_test.target

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
dfV = vectorizer.fit_transform(df)





import random
import math
from numpy.random import permutation

random_indices = permutation(df.index)
test_cutoff = math.floor(len(df)/3)

test = df.loc[random_indices[1:test_cutoff]]
train = df.loc[random_indices[test_cutoff:]]


y_train, y_test = train.label, test.label

trainV = vectorizer.fit_transform(test)
testV = vectorizer.fit_transform(train)





#y_train = y_train.to_sparse()
#y_test = y_test.to_sparse()

#from sklearn.neighbors import KNeighborsClassifier
#neigh = KNeighborsClassifier(n_neighbors=3)
#neigh.fit(train,y_train) 
'''

'''
valoresU = {}

for cols in x_columns:
    vals = df[cols].unique().tolist()    
    valoresU[cols]=(vals)

def getIndexColum(coluna):
    aux = valoresU[coluna]    
    for a in aux:
        print (a, aux.index(a))
    
    
def getIndexColumRow(coluna, linha):
    aux = valoresU[coluna]    
    for a in aux:
        print (a, aux.index(a))
    

for testes in x_columns:
    getIndexColum(testes)

'''
'''

#>>> from sklearn.neighbors import KNeighborsClassifier
#>>> neigh = KNeighborsClassifier(n_neighbors=3)
#>>> neigh.fit(X, y) 

3>>> print(neigh.predict([[1.1]]))

#>>> print(neigh.predict_proba([[0.9]]))

y_column = train["label"]

# The columns that we will be making predictions with.
x_columns = z #['cap', 'ini', 'label', 'next2Cap', 'next2T', 'next2W', 'nextCap', 'nextT', 'nextW', 'palpite', 'prev2Cap', 'prev2T', 'prev2W', 'prevCap', 'prevT', 'prevW', 'simb',  'tipo', 'word'] #['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']
# The column that we want to predict.
y_column = vectorizer.fit_transform(df["label"])

from sklearn.neighbors import KNeighborsRegressor
# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=5)
# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])
'''



'''
import random
import math
from numpy.random import permutation
import numpy as np

pd.DataFrame(df).fillna(0)
nba = df


# Randomly shuffle the index of nba.
random_indices = permutation(nba.index)
# Set a cutoff for how many items we want in the test set (in this case 1/3 of the items)
test_cutoff = math.floor(len(nba)/3)

#np.any(np.isnan(test_cutoff))
# Generate the test set by taking the first 1/3 of the randomly shuffled indices.
test = nba.loc[random_indices[1:test_cutoff]]

# Generate the train set with the rest of the data.
train = nba.loc[random_indices[test_cutoff:]]








# The columns that we will be making predictions with.
x_columns = z #['cap', 'ini', 'label', 'next2Cap', 'next2T', 'next2W', 'nextCap', 'nextT', 'nextW', 'palpite', 'prev2Cap', 'prev2T', 'prev2W', 'prevCap', 'prevT', 'prevW', 'simb',  'tipo', 'word'] #['age', 'g', 'gs', 'mp', 'fg', 'fga', 'fg.', 'x3p', 'x3pa', 'x3p.', 'x2p', 'x2pa', 'x2p.', 'efg.', 'ft', 'fta', 'ft.', 'orb', 'drb', 'trb', 'ast', 'stl', 'blk', 'tov', 'pf']
# The column that we want to predict.
y_column = df["label"]

from sklearn.neighbors import KNeighborsRegressor
# Create the knn model.
# Look at the five closest neighbors.
knn = KNeighborsRegressor(n_neighbors=5)
# Fit the model on the training data.
knn.fit(train[x_columns], train[y_column])
# Make point predictions on the test set using the fit model.
predictions = knn.predict(test[x_columns])

'''


'''
from time import time
from sklearn import metrics
from sklearn.utils.extmath import density
from sklearn.neighbors import KNeighborsClassifier

def benchmark(clf):
    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    score = metrics.accuracy_score(y_test, pred)
    print("accuracy:   %0.3f" % score)

    
    if hasattr(clf, 'coef_'):
        print("dimensionality: %d" % clf.coef_.shape[1])
        print("density: %f" % density(clf.coef_))

        #if opts.print_top10 and x_columns is not None:
        #    print("top 10 keywords per class:")
        #    for i, category in enumerate(categories):
        #        top10 = np.argsort(clf.coef_[i])[-10:]
        #        print(trim("%s: %s"
        #              % (category, " ".join(x_columns[top10]))))
        #print()

    if opts.print_report:
        print("classification report:")
        print(metrics.classification_report(y_test, pred,
                                            target_names=categories))

    if opts.print_cm:
        print("confusion matrix:")
        print(metrics.confusion_matrix(y_test, pred))

    print()
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time
    
benchmark((KNeighborsClassifier(n_neighbors=10), "kNN"))
'''