# AllSpark
Código fuente de la Plataforma de evaluación de Redes Neuronales para Procesamiento de Lenguaje Natural basados en Transformer.</br>
## Instalación. 
Se necesita Python3-pip instalado previamente.</br>

Desde el directorio raíz:</br>
    sudo apt-get install nodejs npm </br>
    npm install --save-dev electron </br>
    npm install is-reachable </br>
Para instalación de las dependencias, usar installer.sh.</br>

Iniciar con npm start.

## Abstract
"NLP's ImageNet moment has arrived", with these words Sebastian Ruder pointed out not only the transcendence of the advances that Deep Learning has catalyzed in all scopes of Machine Learning and Artificial Intelligence in general, Artifical Vision in particular; even more, he showcases that, the impact and this kind of deep changes that it enabled, are repeating themselves within the last years in the field of Natural Language Processing (NLP). The computing power exponential growth and, especially, the advent of a new model, known as Transformer, has changed the state-of-the-art landscape, formerly dominated by handcrafted explicit grammatical and semantic modeling, at first, Shallow Neural Networks at a later stage; based on obtaining word static vector representations from text corpora (word embeddings).

One of the main features, playing a big role in these advances, is known as Transfer Learning, referring to the ability to use already acquired knowledge to learn a new task that is related to this knowledge. With this, Transformer models can be "pre-trained" over massive volumes of unlabeled data (unsupervised learning) in a generic way and, subsequently in each particular case, put it through a "fine-tuning" process that adjusts it to the desired specific task, with a far less amount of data and computing power needed. This has incentivised the popularisation of NLP, by allowing the posibility of achieving state-of-the-art results at a fraction of the computational cost.

The magnitude of this new data driven method success is only equal to the amount of unknowns about how it manages to capture latent semantic relations in the processed text corpus. In this regard, there are numerous studies analysing the word contextual vector representations and, less commonly, the next step of how to combine these word embeddings to provide sentence vector representations (sentence embeddings) that retain the semantic information that the word embeddings carry; process known as semantic composition.

In this scope, this work presents the development of a server-client application capable of obtaining contextual word vectors from a wide variety of Transformer models (from the HuggingFace repository), operate with them to achieve semantic composition through several methods (sum, average, CLS, Information-theoretic Distributional Compositional Semantics parametric function with different parameters) to, lastly, visualise them in a 3-dimensional space using different dimensionality reduction approaches (PCA, tSNE, UMAP) and obtain semantic similarity (cosine similarity, ICM); all of this, while exploiting the hardware resources available.
