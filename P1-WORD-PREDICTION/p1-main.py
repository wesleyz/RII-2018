#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 19:08:30 2018

@author: wesleyz
"""

ColecProb = {}

def bigramation(texto):
    lstTreino = treino.split(" ")
    bigramas = {}
    for i in range(0, len(lstTreino)-1):
        #print(i, lstTreino[i])
        bigramAux = (lstTreino[i], lstTreino[i+1])
        bigramas[bigramAux] = bigramas.get(bigramAux,0)+1
    return bigramas

'''
1) Recebe a palavra e a coleção. 
2) Verifica a palavra na primeira posição da tupla. 
3) Incrementa o n para calcular a probabilidade da proxima tupla
'''
def ProximaPalavraO(frase, colecao):
    auxFrase = frase.split(" ")
    
    frase = auxFrase[len(auxFrase)-1]    
    n = 0
    populacao = 0
    colecao = bigramation(colecao)
    
    for i in colecao.values():
        populacao = populacao + i
          
    for i in colecao.keys():        
        #if bigrama == i[0] or bigrama == i[1] :        
        if frase == i[0] :
            n = n + 1 
            aux = {}
            aux[i] =       colecao[i] / populacao          
            ColecProb[i] = aux[i]
            aux.clear()
    if len(ColecProb) != 0:
        maisprovavel = max(ColecProb.values())
        word = list(ColecProb.keys())[list(ColecProb.values()).index(maisprovavel)]
        return word[1]

#Setups = {}

treino = "I like photograpy I like science I love math"
testes = ['I', 'I like', 'science I love']
set1 = (treino, testes)


treino = "ser ou não ser eis a questão"
testes = ['ser', 'ser eis', 'ser eis a']
set2 = (treino, testes) 


treino = "O desenvolvimento do raciocínio logico é uma importante área da construção de um pensamento crítico sobre fatos e relações que percebemos no \
mundo este cenário de aprendizado tem se tornado desafiador para professores"
testes = ['este', 'tem se', 'sobre os fatos e']
set3 = (treino, testes) 


setup = [set1, set2, set3]

for config in setup:
    treino = config[0]
    testes = config[1]
    print("##############################################")
    print("Texto de Treino: %s \n" % treino)
    for t in testes:
        print('Frase teste:.................. %s'% t)
        print('Palavra predita:..............[%s]\n' % ProximaPalavraO(t, treino))
        ColecProb.clear()