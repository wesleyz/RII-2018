#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:02:09 2019

@author: wesleyz
"""

import os
path = os.getcwd() 




# parse an xml file by name

file = path + '/TesteMarcos_anotado.xml' 

file = '/media/wesleyz/HD1Tera/DADOS/000-git/004-DATASETS/000 dataset portarias ifes/dataset-portarias-ifes/NER-50-portarias/ArqsTeste/NER-50-portarias.xml'


import xml.etree.ElementTree as ET  
tree = ET.parse(file)  
root = tree.getroot()

# one specific item attribute
print('Item #2 attribute:')  
print(root[0][1].attrib)

# all item attributes
labels = []
print('\nAll attributes:')  
for elem in root:  
    
    doc = elem.attrib['DOCID']
    print(doc)
    for subelem in elem:
        
        print('Word', subelem.text)
        print('Features', subelem.attrib)
        tipo = subelem.attrib['CATEG']
    
        feat = subelem.text
        labels.append((doc, feat, tipo))
        


topo = "<html><head><link rel='stylesheet' type='text/css' href='http://rii.lcad.inf.ufes.br/wpsilvaExperimentos/repo-meia/estilo.css'><meta charset='UTF-8'><link rel='icon' type='image/png' href='./logo-lcad.png' /><title>Diário de Doutoramento - PPGI/UFes</title></head><style>    table {    border-spacing: 0;    width: 100%;    border: 0px solid #ddd;    }    th {    cursor: pointer;    }    th, td {    text-align: left;    padding: 16px;    }    tr:nth-child(even) {       background-color: #f2f2f2    }     h2 {  color: darkblue;   font-family: verdana;   font-size: 100%;  }   p{   color: black;  font-family: courier;   font-size: 100%; }     </style> <body><h1 align='center'> <img src='http://rii.lcad.inf.ufes.br/wpsilvaExperimentos/portarias/ifes-portarias-61933-indice-html/logo-ufes.png' width='100' height='100'><img src='http://rii.lcad.inf.ufes.br/wpsilvaExperimentos/portarias/ifes-portarias-61933-indice-html//logo-lcad.png'  width='100' height='100'></h1> <table width='70%'"
rodape = '</table></body></html>'        

inLinha="<tr><td>"
midLinha="</td><td>"
fimLinha="</td></tr>"



def SaveColecao(colecao):
    listaFeatures = []
    listaDocumentos = []
    with open("NER-50-portarias.html",encoding='utf-8', mode="w+") as file:
        file.write("%s\n" % str(topo))
        #writer = csv.writer(file, delimiter=" ")        
        #writer.writerow(topo)
        for i in range(0, len(colecao)):
            
            doc = colecao[i]            
            filename = doc[0]
            paginas = doc[1]
            feat3 = doc[2]
            listaFeatures.append(feat3)                        
            ln = ["<a href='http://rii.lcad.inf.ufes.br/wpsilvaExperimentos/portarias/ifes-portarias-61933-indice-html/",filename, ".txt' target='_blank'>" ,filename , '</a><br>']            
            link = ''.join(ln)
            linha = (inLinha, filename,midLinha, paginas,midLinha, feat3,midLinha, link, fimLinha)            
            
            file.write(''.join(linha))
        file.write(rodape)
    file.close()
    return listaFeatures, listaDocumentos
    
    

features, documentos = SaveColecao(labels)


counts = dict()
for i in features:
  counts[i] = counts.get(i, 0) + 1





with open('NER-Sumarizacao.csv', 'w') as f:
    for key in counts.keys():
        f.write("%s,%s\n"%(key,counts[key]))








