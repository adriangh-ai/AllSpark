#!/bin/bash
##TEMPORAL##
python3 -m venv $HOME/AllSpark_data/AllSpark_venv
source $HOME/AllSpark_data/AllSpark_venv/bin/activate
pip3 install -U wheel torch transformers numpy==1.20 plotly scikit-learn pandas psutil nltk datasets dash grpcio 
pip3 install -U grpcio-tools waitress dash-uploader dash-daq umap-learn sentencepiece protobuf gensim protobuf3-to-dict
python3 -m nltk.downloader punkt
deactivate
