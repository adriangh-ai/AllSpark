from transformers import AutoModel, AutoTokenizer
import torch

class Basemodel():
    """
    Wrapper parent class meant as an abstraction for actions that are normally performed 
    together: tokens and model to the same device, data parallelisation and tokens to the 
    main device.
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


