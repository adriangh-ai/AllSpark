#!/bin/sh
##TEMPORAL##
#pip3 install --upgrade protobuf
#python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. compservice.proto
#apt-get nodejs, npm
#npm electron
#npm install is-reachable
#npm init, install electron

pip3 install -U torch transformers numpy==1.20 plotly scikit-learn pandas psutil nltk datasets dash grpcio grpcio-tools waitress dash-uploader dash-daq umap-learn sentencepiece protobuf
python3 -m nltk.downloader punkt
