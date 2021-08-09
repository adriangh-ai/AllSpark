import grpc
from src.grpc_files import compservice_pb2, compservice_pb2_grpc

class Server_grpc_if():
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = compservice_pb2_grpc.compserviceStub(self.channel)
    def downloadModel(self, model):
        return self.stub.downloadModel(compservice_pb2.Model(modelname=model)) 
    def deleteModel(self, model):
        return self.stub.deleteModel(compservice_pb2.Model(modelname=model)) 
    def getModels(self):
        _response = self.stub.getModels(compservice_pb2.Empty(empty=0)).model
        _response = {i.name : {'layers':i.layers, 'size':i.size} for i in _response}
        return _response

    def updloadDataset(self, Dataset):
        pass 
    def deleteDataset(self, DatasetName):
        pass 
    def getDatasets(self, Empty):
        pass 

    def inf_session(self, Session):
        return self.stub.inf_session(Session)
    def getDevices(self):
        device_list = self.stub.getDevices(compservice_pb2.Empty(empty=0))
        for device in device_list.dev:
            yield device

