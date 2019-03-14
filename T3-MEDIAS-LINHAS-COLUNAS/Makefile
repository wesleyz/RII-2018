# Shell to use with Make
SHELL := /bin/bash

### Trabalho da Semana 1 - RII
### Aluno: Wesley Pereira da Silva


# Set important Paths
PROJECT := 'Semana 2 - RI'
LOCALPATH := $(CURDIR)/$(PROJECT)
PYTHONPATH := $(LOCALPATH)/


help:
	@echo "    executa"
	@echo "        Processa o arquivo arquivosaida no diretorio corrente."

all: 	compila executa


compila:	executa clear


executa:
	Rscript Westrix.r
	tr '\n' ' ' < column-mean.dat > file.tmp
	mv file.tmp column-mean.dat
clear:
	rm column-mean.dat row-mean.dat
