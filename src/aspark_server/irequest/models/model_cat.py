from .basemodel import Basemodel
from pathlib import Path
import json
from abc import ABC, abstractmethod

class Model_factory(ABC):
    """
    Factory class. Loads the config file an instantiates a model object based on model_type.
    """
    def get_model(modelname, devices):
        
        with open(Path(f'{modelname}/config.json'), 'r') as data_file:
                data = json.load(data_file)
                data = data['model_type']
        
        encodeco_rules = [      #Encoder-Decoder Models
            "ByT5" in data,
            'marian' in data,
            'mt5' in data,
            'pegasus' in data,
            'prophetnet' in data,
            't5' in data,
            'xlm-prophetnet' in data,
            'blenderbot' in data
        ]

        no_pad_rules = [
            'gpt' in data,
            'transfor-xl' in data
        ]

        if any(encodeco_rules):
            return EncoderDecoder(modelname, devices)
        else:
            if any(no_pad_rules):
                return StandardNoPad(modelname, devices)
            else:
                return StandardModel(modelname, devices)

class StandardModel(Basemodel):
    """
    Class for the model with default configuration
    """
    def __init__(self, modelname, devices):
        Basemodel.__init__(self, modelname, devices)

class StandardNoPad(Basemodel):
    """
    Class for the models with default configuration, but have not been
    trained with PAD token, like GPT.
    """
    def __init__(self, modelname, devices):
        Basemodel.__init__(self,modelname,devices)
        self.pad = False
    def tokenize(self, batch):
        """
        Tokenize a batch of sentences. Sames as inherited, but without padding and
        truncation.
        """
        _tokens = self.tokenizer(batch
                                ,return_tensors="pt"
                                ,return_special_tokens_mask=True).to(self.devices)
        self.special_mask = _tokens.pop("special_tokens_mask")  #Might change for passing args to inference
        return _tokens
class EncoderDecoder(Basemodel):
    """
    Class for the complete Encoder-Decoder. Takes the Encoder output.
    """
    def __init__(self, modelname, devices):
        Basemodel.__init__(self,modelname,devices)
    def inference(self, tokens):
        return self.model(input_ids=tokens['input_ids'], decoder_input_ids=tokens['input_ids'], output_hidden_states=True)


if __name__ == "__main__":
    model = Model_factory.get_model("bert-base-uncased", ["cpu"])
    print(type(model))
    
    model.to("cuda:1")