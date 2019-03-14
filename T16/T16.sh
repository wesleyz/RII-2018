#!/bin/bash
_base=$1
_processo=$2
_path=`pwd`

wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/cluto/cluto-2.1.2a.tar.gz
wget http://glaros.dtc.umn.edu/gkhome/fetch/sw/cluto/doc2mat-1.0.tar.gz

tar -xvzf ./doc2mat-1.0.tar.gz
tar -xvzf .cluto-2.1.2a.tar.gz


wget http://cs.joensuu.fi/sipu/datasets/pathbased.txt
wget http://cs.joensuu.fi/sipu/datasets/spiral.txt
wget http://cs.joensuu.fi/sipu/datasets/jain.txt
wget http://cs.joensuu.fi/sipu/datasets/flame.txt

wget http://www.inf.ufes.br/~elias/dataSets/aTribuna-21dir.tar.gz
mkdir aTribuna-21dir
mv aTribuna-21dir.tar.gz aTribuna-21dir/
cd aTribuna-21dir
tar -xvzf aTribuna-21dir.tar.gz
for i in `find . | grep txt`; do echo -e "\nFilename: $i"; cat "$i"; done > tmpfile; mv tmpfile all-files.txt;
../doc2mat-1.0/doc2mat -nlskip=1 ./all-files.txt  tribuna21.mat
../cluto-2.1.2/Linux-x86_64/vcluster -sim=dist -clmethod=graph tribuna21.mat 5 >> resposta.txt &

cd ..

../doc2mat-1.0/doc2mat -nlskip=1 ./pathbased.txt  pathbased.txt.mat
../cluto-2.1.2/Linux-x86_64/vcluster -sim=dist -clmethod=graph tribuna21.mat 5 >> resposta.txt &

../doc2mat-1.0/doc2mat -nlskip=1 ./jain.txt  jain.txt.mat
../cluto-2.1.2/Linux-x86_64/vcluster -sim=dist -clmethod=graph tribuna21.mat 5 >> resposta.txt &

../doc2mat-1.0/doc2mat -nlskip=1 ./spiral.txt  spiral.txt.mat
../cluto-2.1.2/Linux-x86_64/vcluster -sim=dist -clmethod=graph tribuna21.mat 5 >> resposta.txt &

../doc2mat-1.0/doc2mat -nlskip=1 ./flame.txt  flame.txt.mat
../cluto-2.1.2/Linux-x86_64/vcluster -sim=dist -clmethod=graph tribuna21.mat 5 >> resposta.txt &








./cluto-2.1.2/
