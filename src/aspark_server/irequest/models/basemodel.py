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
    def _model_instanciation(self):
        model = AutoModel.from_pretrained(self.modelname)
        if not "cpu" in self.devices:
            model = model.to(self.devices[0])
            if len(self.devices)>1:
                model = torch.nn.DataParallel(model, self.devices).share_memory()
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
                                    ,return_special_tokens_mask=True).to(self.devices[0])
        self.special_mask = _tokens.pop("special_tokens_mask")  #Might change for passing args to inference
        return _tokens
    
    def inference(self, tokens):
        return self.model(**tokens, output_hidden_states=True)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self


