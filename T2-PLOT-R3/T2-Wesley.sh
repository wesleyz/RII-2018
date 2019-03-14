#!/bin/bash
_base=$1
_processo=$2
_path=`pwd`



#alfa=`find . -type f \( -name "*.c" \) -exec echo "'{}'" \;`
alfa=`find . -maxdepth 1 -type d  |grep pRiUfes-R`
agente='ProcessaLinks-V2.py'


rm  './observacoes.txt' './erros.txt'
for item in $alfa; do
	#ls -la $item;
	cp $agente $item
	cd $item
	diretorio=`pwd | cut -d'/' -f9`
	#echo ${diretorio%/*}
	diretorio=`echo "["$diretorio"]"`
	python './'$agente
    echo $diretorio >>   ../observacoes.txt

    wc -l observacoes.txt >> '../observacoes.txt'
	sort observacoes.txt >> '../observacoes.txt'
	#echo "Links Invalidos"
	echo $diretorio >> '../erros.txt'
	wc -l erros.txt >> '../erros.txt'
	sort erros.txt >> '../erros.txt'

	echo '--------------------------------------' >> '../observacoes.txt'
	echo '--------------------------------------' >> '../erros.txt'


	rm './'$agente
	cd ..

  #arquivo=`echo $item |tr -d \'\" `;
  #caminho=`realpath $arquivo`;
  #pathCaminho=`echo ${caminho%/*}`;
  #arquivosaida=`echo $pathCaminho'/arquivo.saida'`;
  #plogfile=`echo $pathCaminho'/output.log'`;
  #prologFile=`echo $pathCaminho'/resposta.pl'`;
  #html2text $arquivo > $prologFile;
  #swipl --quiet -q -o $plogfile -c $prologFile 2>> $arquivosaida;
  #swipl --quiet -q  -c $prologFile   &> $arquivosaida;
done
cat './observacoes.txt'
cat './erros.txt'
