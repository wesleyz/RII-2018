import re
import unicodedata
from sklearn.feature_extraction.text import CountVectorizer

def remover_acentuacao(string):
    string = unicodedata.normalize('NFD', string)
    return u''.join(ch for ch in string if unicodedata.category(ch) != 'Mn')

def cleanrow(row):
    '''remocao de stopwords'''
    row = re.sub("<.*?>", "", row)
    tmp = [p.lower() for p in row.split()]
    aux = []
    '''limpeza dos sinais graficos'''
    for s in tmp:
        aux.append(re.sub(r'\W+', '', remover_acentuacao(s)))
    return aux

with open("corpus.csv", 'r') as txt:
    txt_ = txt.read()
    data = [t for t in txt_.split("<top>") if t.strip() != ""]
    data = [d.split("<title>")[1] for d in data]
    data = [d.split("<con>")[0] for d in data]
    data = ([" ".join(cleanrow(d)) for d in data])

vectorizer = CountVectorizer()
count = vectorizer.fit_transform(data)
words = vectorizer.get_feature_names()

print(count, words)