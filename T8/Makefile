# Shell to use with Make
SHELL := /bin/bash

### Colocar os valores de cada arquivo na coluna de uma matriz
### Discplina RII - Aluno: Wesley Pereira da Silva

# Set important Paths
#PROJECT :T8'
LOCALPATH := $(CURDIR)/$(PROJECT)
PYTHONPATH := $(LOCALPATH)/


help:
	@echo "    executa"
	@echo "    Processa o arquivo arquivosaida no diretorio corrente."

all: 	executa


compila:
	awk '{print $1 $4 $5 $7}' access.log > arquivosaida


executa:
	 awk '{print $1 $4 $5 $7}' access.log > arquivosaida



clear:
	rm -Rf arquivosaida
