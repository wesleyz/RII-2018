SHELL := /bin/bash
LOCALPATH := $(CURDIR)/$(PROJECT)
PYTHONPATH := $(LOCALPATH)/


help:
	@echo "    Executa"
	@echo "    Processa o arquivo arquivosaida no diretorio corrente."

all: 	executa

compila:
	g++ knn.cpp -lm

executa:
	./a.out < dataset.txt

clear:
	rm -Rf ./a.out
