#coding: utf-8

import sys
import os
import html2text
import csv
import requests




def GetName(path):
    k = path+'/name.info'    
    with  open(k,"r", encoding='iso-8859-1') as txt:
        nome = txt.read()
    txt.close()
    return nome
        


def getMoodleData(mypath="./",nomeresp="resposta.txt"):
    data = []
    sistema = ["__pycache__"]
    arquivos = [x for x in os.listdir(mypath) if x not in sistema]
    subpastas = [os.path.join(mypath,subdir) for subdir in arquivos if os.path.isdir(os.path.join(mypath,subdir))]
    for sub in subpastas:
        with open(sub+"/"+nomeresp,"r", encoding='iso-8859-1') as txt:
            auxfolder = sub[(sub.rfind("/")+1): ]
            auxsub = html2text.html2text(txt.read(), "iso-8859-1")
            nome = GetName(sub)
            if auxsub.strip() != "":
                data.append([auxfolder,auxsub,nome])
    return data, len(data)

bancoEntrada = {}
bancoAvaliacao = []
entradaConsolida = {}
consolidadoFinal = []
avaliador = []
submit = []
submissao = []
count = 0
db,tam = getMoodleData()

for subm in db:
    bancoEntrada['aluno']=subm[0]
    bancoEntrada['respostas']=subm[1]
    bancoEntrada['nome'] = subm[2].lower()
    bancoAvaliacao.append(bancoEntrada.copy())
    bancoEntrada.clear()
    avaliador.append(subm[0])
    submit.append(subm[1])
        
#print(len(bancoAvaliacao))
    
for s in submit:
    s = s.replace("\n","").replace(" ","")
    submissao.append(s)


def UrlToNome(url):
    url = url.replace("http://repositorio.ufes.br/handle/", "").replace("/", "-")
    return url


def GetLinksFromDic():
    lista = []
    badUrls = []
    dic = {}
    for obs in bancoAvaliacao:    
        endereco = obs['respostas'].replace("\n", "").replace(".ht","ht" )        
        endereco = endereco.split("#")
        for urlAux in endereco:        
            link = urlAux.split("|")    
            if len(link) == 2:
                url = link[0].rstrip()
                try:                
                    r = requests.head(url)
                    if r.status_code == 200:
                        no = (obs['nome'],link[1].lstrip().rstrip(),UrlToNome(url))            
                        lista.append(no)
                except:
                    e = sys.exc_info()[0]
                    #print(e)
                    erro = (obs['nome'],url)
                    badUrls.append(erro)
            else:
                erro = (obs['nome'].lower(),urlAux)
                badUrls.append(erro)

                #link = link.replace("#", "").replace("\n", "").replace(".", "")
    return lista, badUrls
    

def SaveData(data, filename):
    with open(filename,encoding='utf-8', mode="w+") as file:        
        writer = csv.writer(file, delimiter=";")
        for i in data:
            writer.writerow(i)
    file.close()

observacoes, erros = GetLinksFromDic()

SaveData(erros, 'erros.txt')
SaveData(observacoes, 'observacoes.txt')


    

                     

        
        
