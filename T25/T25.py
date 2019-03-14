#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 20:39:58 2018

@author: wesleyz
"""

import powerlaw
import numpy


def Load_data(namefile):    
    colecao  = []
    documento = {}
    file =  namefile #'/arquivosaida'
    
    with open(file, 'r', encoding='ISO-8859-1') as infile:
        for line in infile:
            #valor = line.read().replace(' ','').decode("ISO-8859-1").split('\n')
            valor = line.replace('\n','').split(" ")
            word = valor[1].strip()
            count = valor[2].strip()
            documento['palavra'] = word
            documento['contagem'] = count            
            aux = documento.copy()
            colecao.append(aux)
            documento.clear()
    infile.close()
    return colecao

dados = Load_data('dictionary.txt')

d = []
d.sort()

for i in dados:
    d.append(int(i['contagem']))


#d=[6, 4, 0, 0, 0, 0, 0, 1, 3, 1, 0, 3, 3, 0, 0, 0, 0, 1, 1, 0, 0, 0, 3,2,  3, 3, 2, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 2, 1, 0, 1, 0, 0, 0, 0, 1,0, 1, 2, 0, 0, 0, 2, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1,3, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, 2, 3, 2, 1, 0, 0, 0, 1, 2]
fit = powerlaw.Fit(numpy.array(d)+1,xmin=1,discrete=True)
fit.power_law.plot_pdf( color= 'b',linestyle='--',label='fit ccdf')
fit.plot_pdf( color= 'b')

print('alpha= ',fit.power_law.alpha,'  sigma= ',fit.power_law.sigma)