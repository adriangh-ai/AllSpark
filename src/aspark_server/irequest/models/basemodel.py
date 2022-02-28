from abc import ABC, abstractmethod

from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np

import gensim.downloader as api
from nltk import tokenize

class Model(ABC):
    """
    Abstract class declaring methods required for subclasses
    """
    def __init__(self):
        pass
    @abstractmethod
    def _model_instanciation(self):
        pass
    @abstractmethod
    def tokenize(self):
        pass
    @abstractmethod
    def inference(self):
        pass
    @abstractmethod
    def inference(self):
        pass

class Basemodel(Model):
    """
    Wrapper parent class for Transformer models, meant as an abstraction for actions that 
    are normally performed together: tokens and model to the same device, data parallelisation 
    and tokens to the main device.
    Relies on children for specifics on how to tokenize and perform inference.
    """
    def __init__(self, modelname, devices):
        self.modelname=modelname
        self.devices = devices
        self.tokenizer=AutoTokenizer.from_pretrained(modelname)
        self.model= self._model_instanciation()
        self.special_mask = None
        self.pad = True #Pad default

    def _model_instanciation(self):
        """
        Creates the model object from model file. If the computing device is not the
        CPU, it sends the model to the selected device's memory.
        """
        model = AutoModel.from_pretrained(self.modelname)
        if not "cpu" in self.devices:
            model = model.to(self.devices)
        #    if len(self.devices)>1:
        #        model = torch.nn.DataParallel(model, self.devices).share_memory()
        return model.eval() 

    def tokenize(self, batch):
        """
        Tokenize a batch of sentences. Adds the mask of the special_tokens in the tokenized sentence.
        This mask is popped, so we can save the value but still gets removed from the token dictionary,
        as huggingface models don't support the passing of this parameter to the model
        """
        _tokens = self.tokenizer(batch, padding=True
                                    ,truncation=True
                                    ,return_tensors="pt"
                                    ,return_special_tokens_mask=True).to(self.devices)
        self.special_mask = _tokens.pop("special_tokens_mask")  #Might change for passing args to inference
        return _tokens
    
    def inference(self, tokens):
        """
        Sends the batch of tokens to the model for the feed forward pass.
        """
        return self.model(**tokens, output_hidden_states=True)
    
    def to(self, device):
        """
        Sends model to device's memory.
        """
        self.model = self.model.to(device)
        return self
    
    def paddding(self):
        """
        Getter for pad
        """
        return self.pad

class Base_Staticmodel(Model):
    """
    Parent class for models with static representations as outputs
    """
    def __init__(self, modelname):
        self.modelname = modelname
        self.model= self._model_instanciation()
        self.special_mask= None
    
    def _model_instanciation(self):
        """
        Loads the Gensim Model
        """
        return api.load(self.modelname)
   
    def tokenize(self, sentences):
        """
        Uses NLTK to tokenize the sentences into words. If the model is uncased,
        lower() function is used to lower case the input.
        """
        _cased = True
        if 'uncased' in api.info()['models'][self.modelname].get('base_dataset', ''):
            _cased = False
        
        return [tokenize.word_tokenize(i if _cased else i.lower()) for i in sentences]

    
    def inference(self, sentences):
        """
        Goes through the list of words of each sentence obtaining the static embedding.
        Checks if the word is in the model dictionary before getting the embedding. 
        """
        _embeddings = []
        for sentence in sentences:
            _sentence_embedding = []
            for word in sentence:
                if self.model.__contains__(word):       # Guard. Is the word in the model?
                    _sentence_embedding.append(self.model.get_vector(word))
            _sentence_embedding = np.array([_sentence_embedding])
            _embeddings.append(_sentence_embedding)
        return [torch.from_numpy(i) for i in _embeddings]
   
if __name__ == "__main__":
    #TEST
    model = Base_Staticmodel('fasttext-wiki-news-subwords-300')
    tokens = model.tokenize(['This is a test.', 'Test.', 'Testing this.'])
    print(tokens)
    output = model.inference(tokens)
    _output = torch.stack([torch.sum(i,1) for i in output])
    _output = torch.mean(_output,1)
    print(_output)
    _output = [ptensor.detach().numpy() for ptensor in _output]
    import pandas as pd
    _output = pd.DataFrame(_output)
  