# gdown https://drive.google.com/uc?id=1hxE0OPR7VLEHesr48WisynuoNMhXJbpl &&
# unzip data.zip &&
#!/bin/bash
if [ -e dataset/ogbn_arxiv_orig/titleabs.tsv ]
then
    echo "Arxiv dataset already exists."
else
    wget https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz
    gzip -d titleabs.tsv.gz
    mkdir dataset
    mkdir dataset/ogbn_arxiv_orig
    mv titleabs.tsv dataset/ogbn_arxiv_orig
fi