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
        

        
import csv
def SaveColecao(colecao):
    with open("labels-extraidos-48-portarias.csv",encoding='utf-8', mode="w+") as file:
        writer = csv.writer(file, delimiter=";")
        for i in range(0, len(colecao)):
            doc = colecao[i]
            filename = doc[0]
            paginas = doc[1]
            feat3 = doc[2]            
            linha = (filename, paginas, feat3)            
            print(linha)
            writer.writerow(linha)
    file.close()
    
    

SaveColecao(labels)

import pandas as pd


L = ['Thanks You', 'Its fine no problem', 'Are you sure']


'''
#create new df 
dfLabel = pd.DataFrame(labels)
dfLabel.colnames(dfLabel)[0] <- "feat"
dfLabel.colnames(dfLabel)[1] <- "label"
print (dfLabel)


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
dfLabel["make_code"] = lb_make.fit_transform(obj_df["make"])
obj_df[["make", "make_code"]].head(0)

'''

# one specific item's data
#print('\nItem #2 data:')  
#print(root[0][1].text)

# all items data
'''        
print('\nAll item data:')  
for elem in root:  
    for subelem in elem:
        print(subelem.text)
'''