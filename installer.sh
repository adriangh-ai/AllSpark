#!/bin/sh
##TEMPORAL##
apt-get update
apt-get upgrade
apt-get -y install python3-pip
#nltk en vez de spacy, para sentences es más rápido no carga un modelo de parser ni hace arbol
pip3 install -U torch transformers numpy plotly==5.1.0 scikit-learn pandas psutil nltk datasets
python3 -m nltk.downloader punkt