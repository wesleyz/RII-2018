# Shell to use with Make
SHELL := /bin/bash

### P-1 Proponha uma estratégia para predição da próxima palavra pn, dado uma
### sequência de N outras palavras anteriores {p0, p1,..., pn−1}. Suponha, inicialmente,
### que N ≤ 3. Submeta seus códigos, dados e makefile necessários para testes em uma máquina Linux.a
### Discplina RII - Aluno: Wesley Pereira da Silva

# Set important Paths
#PROJECT := 'Semana 4 - RI'
LOCALPATH := $(CURDIR)/$(PROJECT)
PYTHONPATH := $(LOCALPATH)/


help:
	@echo "    executa"
	@echo "    Processa o arquivo arquivosaida no diretorio corrente."
	
all: 	executa


compila:	
	python p1-main.py
	

executa:
	python p1-main.py > arquivosaida
	


clear:
	rm -Rf arquivosaida