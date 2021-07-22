from transformers import AutoModel, AutoTokenizer

class Basemodel():
    def __init__(self, modelname):
        self.modelname=modelname
        self.tokenizer=AutoTokenizer.from_pretrained(modelname)
        self.model= AutoModel.from_pretrained(modelname).eval()
        self.special_ids= self.tokenizer.all_special_ids
    
    def tokenize(self, batch):
        return self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
    
    def inference(self, tokens):
        return self.model(**tokens, output_hidden_states=True)
    
    def to(self, device):
        self.model = self.model.to(device)
        return self


