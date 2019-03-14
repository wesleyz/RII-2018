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


compila:
	awk '{print $1 $4 $5 $7}' access.log > arquivosaida


executa:
	 python3 google2.py 'Quem descobriu o Brasil' > arquivosaida
	 ./GetFiles-Links.sh




clear:
	rm -Rf datasetLinks
