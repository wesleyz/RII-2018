# Shell to use with Make
SHELL := /bin/bash

###
### Discplina RII - Aluno: Wesley Pereira da Silva

# Set important Paths
#PROJECT :14'
LOCALPATH := $(CURDIR)/$(PROJECT)
PYTHONPATH := $(LOCALPATH)/


help:
	@echo "    executa"
	@echo "    Processa o arquivo arquivosaida no diretorio corrente."

all: 	executa


compila: 	executa



executa:
	python3 aLine-Information-Retrieval.py
	cut -d';' -f 1 brasil > arquivosaida
	./GetFiles-Links.sh
	cut -d';' -f 1 brazil > arquivosaida
	./GetFiles-Links.sh





clear:
	rm -Rf datasetLinks
	rm arquivosaida brasil brazil
