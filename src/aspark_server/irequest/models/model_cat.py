from .basemodel import Basemodel

class Model_factory():
    def get_model(modelname, devices):
        if "bert" in modelname:
            return dummy(modelname, devices)

class dummy(Basemodel):
    def __init__(self, modelname, devices):
        Basemodel.__init__(self, modelname, devices)

if __name__ == "__main__":
    model = Model_factory.get_model("bert-base-uncased", ["cpu"])
    print(type(model))
    
    model.to("cuda:1")