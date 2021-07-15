#!/bin/sh
##TEMPORAL##
apt-get update
apt-get upgrade
apt-get -y install python3-pip
pip3 install torch transformers numpy plotly==5.1.0 scikit-learn pandas psutil