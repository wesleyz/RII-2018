#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 07:11:02 2018

@author: wesleyz
"""

import numpy as np
from os import listdir
from os.path import isfile, join
import os



path = os.getcwd()
isDown = os.path.isdir(path+'/dataset/')
pathFiles = path + '/metricas'

os.system('wget http://moodle.lcad.inf.ufes.br/enunciados/metricas.tar.gz')
os.system('tar -xzf ./metricas.tar.gz')
print ('Extracting data.')
os.system('rm -rf metricas.tar.gz')
print ('Dataset ready.')


vetor = np.zeros(shape=(37,17))
contador = 0


onlyfiles = [f for f in listdir(".")
                  if isfile(join(".", f))
            ]

for arq in onlyfiles:
   if arq.endswith(".dat"):
       vetor[contador]= np.loadtxt(arq, dtype=float)
       print (contador, arq)
       #listaArquivos[contador] = arq.toString()
       contador = contador + 1


np.savetxt("arquivosaida",  vetor, delimiter=';', fmt='%1.4f')
os.system('rm *.dat')
print ('Clean work data.')

print("Resultado do processamento no arquivosaida no diretorio corrente.")
