# Shell to use with Make
SHELL := /bin/bash

###
### Discplina RII - Aluno: Wesley Pereira da Silva

# Set important Paths
#PROJECT :15'
LOCALPATH := $(CURDIR)/$(PROJECT)
PYTHONPATH := $(LOCALPATH)/


help:
	@echo "    executa"
	@echo "    Processa o arquivo arquivosaida no diretorio corrente."

all: 	executa


compila: 	executa



executa:
	python3 T15.py > arquivosaida
	grep -v 'Step' arquivosaida > resposta.txt
	./macros.sh > macros.tex
	pdflatex relatorio.tex


clear:
	rm  arquivosaida Relatorio.pdf
