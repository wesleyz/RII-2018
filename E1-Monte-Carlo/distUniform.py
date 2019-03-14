#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 18:37:21 2018

@author: wesleyz
"""

from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


'''
fig, ax = plt.subplots(1, 1)


mean, var, skew, kurt = norm.stats(moments='mvsk')

x = np.linspace(norm.ppf(0.01),
                norm.ppf(0.99), 100)
ax.plot(x, norm.pdf(x),
       'r-', lw=5, alpha=0.6, label='norm pdf')
'''

mu, sigma = 0, 1 # mean and standard deviation
s = np.random.uniform(mu, sigma, 100)
k = pd.DataFrame(s)

print (s)





import matplotlib.pyplot as plt
count, bins, ignored = plt.hist(s, 30, normed=True)
plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),  linewidth=2, color='r')
plt.show()


print (k.describe())

