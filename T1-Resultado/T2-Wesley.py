#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:59:21 2018

@author: wesleyz
"""

import numpy as np
import os
import math
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D



def getDimVec (path):    
    with open(path,'r') as f:        
        dim = int(f.readlines()[1].split(':')[1])        
    return dim
    
def getVectors (path, dim, s):
    feat = np.loadtxt(path,skiprows=2,dtype=np.int32)
    vecs = np.zeros((s,dim))       
    for f in feat:
        vecs[f[0]-1, f[1]-1] = f[2]            
    return vecs


dimVec = getDimVec ('dataset.conf')
#dimVec = getDimVec ('dataset.conf')
x = []
y = []
z = []

vecs = getVectors ('features.mtx', dimVec, 4)
for i in vecs:
    x.append(i[0])
    y.append(i[1])
    z.append(i[2])

fig = pyplot.figure()
ax = Axes3D(fig)
#ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c = 'b', marker='o')
ax.scatter(x,y,z, c = 'b', marker='o')
ax.set_xlabel('Feat X')
ax.set_ylabel('Feat Y')
ax.set_zlabel('Feat Z')

pyplot.show()