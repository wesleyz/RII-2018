#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 24 20:12:39 2018

@author: wesleyz
"""





import pandas as pd
df = pd.read_csv('dataset100.csv', delimiter=';')
df.head()
df.columns = ['documento','texto']



from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(df['texto'])

#print(df)

X = tfidf.transform(df['texto'])
z = X.todense(order=None, out=None)


import csv

with open("dataset100.csv") as csvfileIn:
    reader = csv.reader(csvfileIn, delimiter=';') 
    all_rows= list(reader)


#print(all_rows)






'''
from numpy import linalg

u,s,vt = linalg.svd(z)
print('Shape X', )
print(X.shape)
print('Shape U', u.shape)
print('Print u\n', u)


print('Shape s\n', s)
print( s.shape)
print('Print vt\n', vt)
print(vt.shape)
'''


import pickle
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
import csv

import numpy as np
from math import sqrt
from math import log
import matplotlib.pyplot as plt



###############################################################################
#  Load the raw text dataset.
###############################################################################

print("Loading dataset...")

# The raw text dataset is stored as tuple in the form:
# (X_train_raw, y_train_raw, X_test_raw, y_test)
# The 'filtered' dataset excludes any articles that we failed to retrieve
# fingerprints for.

qtdLinhaTreino = 3
qtdLinhaTeste = 1

X_train_raw = []
y_train_labels = []
X_test_raw = []
y_test_labels =[]

i = 0
raw_text_dataset = all_rows #pickle.load( open( "dataset5.pickle", "rb" ) )
for i in range(0,qtdLinhaTreino):
    X_train_raw.append(all_rows[i][1])  #[all_rows[i][1], all_rows[1][1], all_rows[2][1]]#raw_text_dataset[0]
    y_train_labels.append(all_rows[i][0])  #[all_rows[0][0], all_rows[1][0], all_rows[2][0]]

indice = qtdLinhaTreino
fimIndice = indice+qtdLinhaTeste
for j in range(indice,fimIndice):
    X_test_raw.append(all_rows[j][1])  #[all_rows[4][1], all_rows[5][1]] #, all_rows[2][1]]
    y_test_labels.append(all_rows[j][0])  #[all_rows[4][1], all_rows[5][1]] #raw_text_dataset[3]

# The Reuters dataset consists of ~100 categories. However, we are going to
# simplify this to a binary classification problem. The 'positive class' will
# be the articles related to "acquisitions" (or "acq" in the dataset). All
# other articles will be negative.
y_train = ["acq" in y for y in y_train_labels]
y_test = ["acq" in y for y in y_test_labels]




print("  %d training examples (%d positive)" % (len(y_train), sum(y_train)))
print("  %d test examples (%d positive)" % (len(y_test), sum(y_test)))


###############################################################################
#  Use LSA to vectorize the articles.
###############################################################################

# Tfidf vectorizer:
#   - Strips out “stop words”
#   - Filters out terms that occur in more than half of the docs (max_df=0.5)
#   - Filters out terms that occur in only one document (min_df=2).
#   - Selects the 10,000 most frequently occuring words in the corpus.
#   - Normalizes the vector (L2 norm of 1.0) to normalize the effect of 
#     document length on the tf-idf values. 
vectorizer = TfidfVectorizer() #(max_df=1, max_features=10000, min_df=2, stop_words='english',                              use_idf=True)

# Build the tfidf vectorizer from the training data ("fit"), and apply it 
# ("transform").
X_train_tfidf = vectorizer.fit_transform(X_train_raw)



print("  Actual number of tfidf features: %d" % X_train_tfidf.get_shape()[1])

print("\nPerforming dimensionality reduction using LSA")
t0 = time.time()

# Project the tfidf vectors onto the first N principal components.
# Though this is significantly fewer features than the original tfidf vector,
# they are stronger features, and the accuracy is higher.
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))

# Run SVD on the training data, then project the training data.
X_train_lsa = lsa.fit_transform(X_train_tfidf)

print("  done in %.3fsec" % (time.time() - t0))

explained_variance = svd.explained_variance_ratio_.sum()
print("  Explained variance of the SVD step: {}%".format(int(explained_variance * 100)))


# Now apply the transformations to the test data as well.
X_test_tfidf = vectorizer.transform(X_test_raw)
X_test_lsa = lsa.transform(X_test_tfidf)
print('TfIdf')
print(X_test_tfidf)
print('LSA')
print(X_test_lsa)


###############################################################################
#  Run classification of the test articles
###############################################################################

print("\nClassifying tfidf vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 
# and brute-force calculation of distances.
knn_tfidf = KNeighborsClassifier(n_neighbors=2, algorithm='brute', metric='cosine')
knn_tfidf.fit(X_train_tfidf, y_train)

# Classify the test vectors.
p = knn_tfidf.predict(X_test_tfidf)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
print("  done in %.3fsec" % elapsed)


print("\nClassifying LSA vectors...")

# Time this step.
t0 = time.time()

# Build a k-NN classifier. Use k = 5 (majority wins), the cosine distance, 
# and brute-force calculation of distances.
knn_lsa = KNeighborsClassifier(n_neighbors=2, algorithm='brute', metric='cosine')
omega = knn_lsa.fit(X_train_lsa, y_train)

# Classify the test vectors.
p = omega.predict(X_test_lsa)

# Measure accuracy
numRight = 0;
for i in range(0,len(p)):
    if p[i] == y_test[i]:
        #print(p[i], y_test[i])
        numRight += 1

print("  (%d / %d) correct - %.2f%%" % (numRight, len(y_test), float(numRight) / float(len(y_test)) * 100.0))

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)    
print("    done in %.3fsec" % elapsed)




#####################################





def multiply(*args):
    """takes custom number of numpy arrays and multiplies them
    Arguments:
    take as many numpy arrays as necessary
    """
    i = 0
    res = 1
    while i < len(args):
        M = args[i]
        i += 1
        res = np.dot(res, M)
    return res


def cosine(x, y):
    """returns cosine of angle between x et y"""
    return np.dot(x, y)/sqrt(np.dot(x, x))/sqrt(np.dot(y, y))


def build_terms(docs, stopwords):
    """build dictionary of words from docs, ignoring list of stopwords """
    terms = list(set([item.lower() for s in docs for item in s.split(" ")
                      if item.lower() not in stopwords]))
    terms.sort()
    terms = dict((key, value) for (value, key) in enumerate(terms))
    return terms


def build_M(terms, docs):
    """take a list of string docs, and a dict for terms
       extracts vector of words and build term-doc matrix"""
    docs_split = [doc_list.lower().split(" ") for doc_list in docs]
    n_docs = len(docs)
    n_terms = len(terms)

    M = np.zeros((n_terms, n_docs))

    for i, doc in enumerate(docs_split):
        for term in doc:
            if term in terms:
                M[terms.get(term), i] += 1
    return M


def tfidf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead
    Arguments:
    M : numpy 2d float array
    """

    return tf(M)*idf(M)


def idf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead
    Arguments:
    M : numpy 2d float array
    """
    n_terms = M.shape[0]
    n_docs = float(M.shape[1])
    Mtfidf = np.zeros((n_terms, n_docs))

    for term in range(n_terms):
        dt = float(np.count_nonzero(M[term]))
        Mtfidf[term] = log(n_docs/dt) if dt != 0 else 0
    return Mtfidf


def tf(M):
    """take matrix term-doc with frequencies and normalize with tf-idf instead
    Arguments:
    M : numpy 2d float array
    """

    n_terms = M.shape[0]
    n_docs = M.shape[1]
    Mtf = np.zeros((n_terms, n_docs))

    for doc in range(n_docs):
        # Mtf[:, doc] = M[:, doc]/M[:, doc].sum()
        Mtf[:, doc] = M[:, doc]/M[:, doc].max()
    return Mtf


def scatter(U, V, labels):
    plt.scatter(U, V)
    [plt.annotate(label, xy=(x, y), xytext=(0.5, 9), textcoords="offset points", ha="right", va="bottom") for label, x, y in zip(labels, U, V)] 


# def main():

# densier numpy array printing
np.set_printoptions(precision=3)

# documents = [
#     "Human machine interface for lab abc computer applications",
#     "A survey of user opinion of computer system response time",
#     "The EPS user interface management system",
#     "System and human system engineering testing of EPS",
#     "Relation of user perceived response time to error measurement",
#     "The generation of random binary unordered trees",
#     "The intersection graph of paths in trees",
#     "Graph minors IV Widths of trees and well quasi ordering",
#     "Graph minors A survey"
# ]

# docs_label = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]

stopwords = set([''])

#==========
documents = X_train_raw #[ 
#   "chat souris",
#    "chat souris animaux",
#    "souris clavier",
#    "clavier ordinateur"
#]
docs_label = ["A1", "A2", "B1", "B2"]  # chat souris etc

terms_label = build_terms(documents, stopwords)

# ==========
# using custom terms
# terms_label = ["human",
#          "interface",
#          "computer",
#          "user",
#          "system",
#          "response",
#          "time",
#          "EPS",
#          "survey",
#          "trees",
#          "graph",
#          "minors"]


#==========
# documents = [
#     "A A A",
#     "B B C A C",
#     "A B A C",
#     "A A A B",
#     "X X Y",
#     "Y Y X Z",
#     "X X Z Z"
# ]

# terms_label = build_terms(documents, stopwords)






terms = dict((key.lower(), value) for (value, key) in enumerate(terms_label))


M = build_M(terms, documents)
# M = tfidf(M)

U, s, V = np.linalg.svd(M)
S = np.zeros(M.shape)

# # S[:s.size, :s.size] = np.diag(s)
S[:s.size, :s.size] = np.diag([k if i < 3 else 0 for (i, k) in enumerate(s)])


print('S\n', S)

print('M\n', M)

print('U\n', U)

print('V\n', V)

scatter(U[:, 0], U[:, 1], terms_label)
#scatter(V[:, 1], V[:, 2], docs_label)

# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# # ax.scatter(U[:, 0], U[:, 1], U[:, 2])

plt.show()

