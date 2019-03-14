#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 08:25:14 2018

@author: wesley
"""
#import time
#start = time.time()

from __future__ import division

import nltk
nltk.download('rslp')



def LoadArquivo(file):    
    colecao  = []    
    with open(file, 'r', encoding="utf-8") as infile:
        for line in infile:
            #valor = line.read().replace(' ','').decode("ISO-8859-1").split('\n')
            valor = line.replace('\n',' ').replace('\x93', ' ').replace('\x94', ' ').replace('=','')
            if (len(valor)>1):
                #print(len(valor))
                colecao.append(valor)           
    infile.close()
    return colecao


def SaveColecao(colecao, file):
    with open(file,encoding='utf-8', mode="w+") as file:
        #writer = csv.writer(file, delimiter="")
        for i in colecao:            
            #linha = (filename, paginas, ano, titulo, label, pA, pB)
            #print(linha)
            file.writelines(i+'\n')
            #writer.writerow(i)
            #doc.clear()
    file.close()


#SaveColecao(all_documents, 'usiel.txt')
all_documents = LoadArquivo('usiel.txt')
from yellowbrick.text import FreqDistVisualizer
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
docs       = vectorizer.fit_transform(all_documents)
features   = vectorizer.get_feature_names()
visualizer = FreqDistVisualizer(features=features)
visualizer.fit(docs)
visualizer.poof()
    




















































