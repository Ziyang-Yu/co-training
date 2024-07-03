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


if [ -e cache/ ]
then 
    echo "cache folder already exists."
else
    mkdir cache
    chmod -R 777 cache/
fi


if [ -e cache/cache_emb.pth ]
then
    echo "cached arxiv emb already exists."
else
    wget https://wzrpr-my.sharepoint.com/:u:/g/personal/pannenetsf_wzrpr_onmicrosoft_com/Edq5Db1RittKsEYK-ZuIW30BGZM9Xi_qS8OvoGJtl_Sy8A\?e\=AXpLbR\&download\=1 -O cache/cache_emb.pth
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
    # wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
    gdown https://drive.google.com/uc?id=1icSC92DMjJHJuksfSMvD8_97XiT3VaBF
    # gunzip titleabs.tsv.gz
    mkdir dataset/ogbn_arxiv_orig
    mv titleabs.tsv dataset/ogbn_arxiv_orig
fi

if [ -e model.pt ]
then
    echo "pretrained GNN model already exists."
else
    # wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
    gdown https://drive.google.com/uc?id=1Qk47siNAw-E7g5OuJrHx_uMTDu4AoPkv
    # gunzip titleabs.tsv.gz
fi

if [ -e arxiv_deberta.pt ]
then
    echo "arxiv_deberta already exists."
else
    # wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
    gdown https://drive.google.com/uc?id=1lBEJTfaWxJJqAlNeecUG19BLef5S0ht5
    # gunzip titleabs.tsv.gz
fi
