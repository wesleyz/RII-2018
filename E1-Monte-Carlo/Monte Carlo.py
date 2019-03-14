from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab
import scipy.stats as stats
import math


def calcZScore(x,mx,xs):
    #print('zscore', x, mx, xs)
    
    z = float(x)-float(mx)/float(xs)
    return z
        

def GeraDistribuicao(mu, sigma, n):
    sample = []
    for i in range(n):
        numero = sigma * np.random.randn() + mu
        sample.append(numero) 
    return sample

#def GeradorMonteCarlo (rodadas):
  #Definição de Média e Desvio Padrão da Distribuição Artificial
  #mu, sigma = 53, 6 # mean and standard deviation
  #rodadas = 100
number_of_walks = 100
colData = []
mu, sigma = 53, 6
  
for walk_length in range (1,31):
    no_transport = 0
    for i in range(number_of_walks):
       #(x, y) = random_walk_2(walk_length)
       distribuicao = GeraDistribuicao(mu,sigma,50)   
       dfDistribuicao = pd.DataFrame(distribuicao)
       #print(dfDistribuicao.describe())
       distribuicao.sort()
       Q1 = dfDistribuicao.quantile(0.25)
       Q3 = dfDistribuicao.quantile(0.75)
       IQR = Q3 - Q1
       colData.append(float(IQR))
       l = colData
       if float(IQR) >= 8.5:
          no_transport += 1
            #print('IQR', IQR)
        #print('Q1', Q1)
        #print('Q3', Q3)
    no_transport_percentage = float(no_transport) / number_of_walks
    print("Passo = ", walk_length, "/ Percentual de alunos com IQR acima de 8.5 = ",100*no_transport_percentage)
    print(no_transport)
    stats.probplot(colData, dist="norm", plot=pylab)
    pylab.show()
    count, bins, ignored = plt.hist(colData, 20, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),  linewidth=2, color='r')
    plt.show()  
    

spD = pd.Series(colData).describe()
sumarioD = pd.Series(colData).describe()
zD = norm.cdf(calcZScore(8.5,spD['mean'],spD['std']/math.sqrt(spD['count'])))
#print(zD)


    
#print(stats.zscore(df, axis=spD['mean'], ddof=spD['std']))

    
#count, bins, ignored = plt.hist(l, 20, normed=True)
#plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ),  linewidth=2, color='r')
#plt.show()    

    #print(len(colData))
    #stats.probplot(colData, dist="norm", plot=pylab)
    #pylab.show()
        
        #dfDistribuicao[dfDistribuicao > 8.5].dropna()
        #IQR = abs(dfDistribuicao.quantile(0.75)) - abs(dfDistribuicao.quantile(0.25))
        #distance = abs(x) + abs(y)        
        #colData.append(IQR)        
        #if distance <= 4:
        #if float(IQR) > abs(dfDistribuicao.quantile(0.5)):     
           #no_transport +=1
           #print(no_transport)
    #no_transport_percentage = float(no_transport) / number_of_walks
    #print("Walk size = ", walk_length, "/ % of no tranport = ",100*no_transport_percentage )
    #stats.probplot(colData, dist="norm", plot=pylab)
    #pylab.show()
  
  
#GeradorMonteCarlo(50)
    
    
    
import random





    
def random_walk_2(n):
    x, y = 0,0
    for i in range(n):
        (dx, dy) = random.choice([(0,1), (0,-1), (1,0),(-1,0)])
        x += dx
        y += dy
    return (x, y)

def run_random_w():
    number_of_walks = 5000
    colData = []    
    for walk_length in range (1,31):
        no_transport = 0
        for i in range(number_of_walks):
            (x, y) = random_walk_2(walk_length)
            distance = abs(x) + abs(y)        
            colData.append(distance)        
            if distance <= 4:
                no_transport +=1
                #print(no_transport)
        no_transport_percentage = float(no_transport) / number_of_walks
        print("Walk size = ", walk_length, "/ % of no tranport = ",100*no_transport_percentage )
        stats.probplot(colData, dist="norm", plot=pylab)
        pylab.show()
