#!/bin/sh
##TEMPORAL##
apt-get update
apt-get upgrade
apt-get -y install python3-pip
#pip3 install --upgrade protobuf
#python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. compservice.proto
#apt-get nodejs, npm
#npm electron
#npm install is-reachable
#waitress
#pip install dash-uploader
#pip install dash-daq
#npm init, install electron
#pip install umap-learn
#pip install sentencepiece
#pip install diskcache multiprocess psutil
#unednlp networkx
pip3 install -U torch transformers numpy plotly==5.1.0 scikit-learn pandas psutil nltk datasets dash grpcio grpcio-tools waitress
python3 -m nltk.downloader punkt