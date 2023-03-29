FROM pytorch/pytorch
WORKDIR /workspaces
COPY . .
RUN apt update && apt -y install git && pip install -U wheel transformers plotly scikit-learn pandas psutil nltk datasets dash grpcio  grpcio-tools waitress dash-uploader dash-daq umap-learn sentencepiece protobuf==3.19 gensim protobuf3-to-dict && python3 -m nltk.downloader punkt
CMD ["init.sh"]
EXPOSE 42000
EXPOSE 42001
