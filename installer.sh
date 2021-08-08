#!/bin/sh
##TEMPORAL##
apt-get update
apt-get upgrade
apt-get -y install python3-pip
#nltk en vez de spacy, para sentences es más rápido no carga un modelo de parser ni hace arbol
#dash, dash-bootstrap-components datasets, grpcio grpc-tools??
#pip3 install --upgrade protobuf
#python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. compservice.proto
#apt-get nodejs, npm
#npm electron
#waitress
#pip install dash-uploader
#pip install dash-daq
#npm init, install electron
pip3 install -U torch transformers numpy plotly==5.1.0 scikit-learn pandas psutil nltk datasets dash grpcio grpcio-tools
python3 -m nltk.downloader punkt