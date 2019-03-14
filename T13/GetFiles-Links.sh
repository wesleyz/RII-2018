#!/bin/bash
_base=$1
_processo=$2
_path=`pwd`

rm -Rf datasetLinks
mkdir datasetLinks
rm indice.dat
cd datasetLinks
while read p; do
  p=`echo $p| tr -d '\r'`
  wget "$p"

done < ../fonte-links2.csv



alfa=`find . -type f \( -name "*.pdf" \) -exec echo "'{}'" \;`
#echo $alfa


for item in $alfa; do
  arquivo=`echo $item |tr -d \'\" `;
  #caminho=`realpath $arquivo`;
  #ls $arquivo
  pdftotext $arquivo
  rm $arquivo
done
cd ..

find datasetLinks/ |grep txt > indice.dat

aLine -i -l indice.dat -d Index-Links-Tribuna
