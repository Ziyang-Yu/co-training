# gdown https://drive.google.com/uc?id=1hxE0OPR7VLEHesr48WisynuoNMhXJbpl &&
# unzip data.zip &&
#!/bin/bash

if [ -e dataset/ ]
then 
    echo "dataset folder already exists."
else
    mkdir dataset
    chmod -R 777 dataset/
fi


if [ -e dataset/arxiv.pt ]
then
    echo "Arxiv embedding already exists."
else
    gdown https://drive.google.com/uc?id=19kfhgsxhJLZn5MZhQaHkdR_yn7qi9Ahd
    mv arxiv.pt dataset/
fi

if [ -e dataset/cora_orig ]
then
    echo "cora orig already exists."
else
    gdown https://drive.google.com/uc?id=1hxE0OPR7VLEHesr48WisynuoNMhXJbpl
    apt-get install unzip
    unzip cora_orig.zip
    mv cora_orig dataset/
fi



if [ -e dataset/ogbn_arxiv_orig/titleabs.tsv ]
then
    echo "Arxiv dataset already exists."
else
    wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
    gzip -d titleabs.tsv.gz
    mkdir dataset/ogbn_arxiv_orig
    mv titleabs.tsv dataset/ogbn_arxiv_orig
fi