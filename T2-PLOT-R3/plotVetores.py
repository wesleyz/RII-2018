#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 18:12:56 2018

@author: wesleyz
"""
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random


fig = pyplot.figure()
ax = Axes3D(fig)

sequence_containing_x_vals = [1,0,0,1]
sequence_containing_y_vals = [0,1,0,1]
sequence_containing_z_vals = [0,0,1,1]

ax.scatter(sequence_containing_x_vals, sequence_containing_y_vals, sequence_containing_z_vals, c = 'b', marker='o')
ax.set_xlabel('Feat X')
ax.set_ylabel('Feat Y')
ax.set_zlabel('Feat Z')

pyplot.show()




