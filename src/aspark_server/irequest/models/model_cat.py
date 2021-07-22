from models.basemodel import Basemodel

class Model_factory():
    def get_model(modelname):
        if "bert" in modelname:
            return dummy(modelname)

class dummy(Basemodel):
    def __init__(self, modelname):
        Basemodel.__init__(self, modelname)

if __name__ == "__main__":
    model = Model_factory.get_model("bert-base-uncased")
    print(type(model))
    
    model.to("cuda:1")