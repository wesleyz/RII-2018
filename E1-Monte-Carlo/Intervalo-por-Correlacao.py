#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 18:44:24 2018

@author: wesleyz
"""

import numpy as np
from scipy import stats

def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    r_z = np.arctanh(r)
    se = 1/np.sqrt(x.size-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    
    return r, p, lo, hi


x = np.random.randint(1, 10, 8)
y = np.random.randint(1, 10, 8)
alfa = pearsonr_ci(x,y)
print('Pearson correlation coefficient-------------------- ',alfa[0])
print('The corresponding p value--------------------------',alfa[1])
print('The lower and upper bound of confidence intervals--',alfa[2])
print('The lower and upper bound of confidence intervals---',alfa[3])

import matplotlib.pyplot as plt

plt.plot(x,y, 'ro')
plt.axis([0, 10, 0, 10])
plt.show()