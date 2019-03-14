#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 11:31:05 2019

@author: wesleyz
"""

from sklearn.feature_extraction.text import CountVectorizer
corpus = ['This is the first document.', 'This document is the second document.', 'And this is the third one.', 'Is this the first document?',           ]   
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())  