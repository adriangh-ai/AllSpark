import grpc
from src.grpc_files import compservice_pb2, compservice_pb2_grpc

class Server_grpc_if():
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address
                                            ,options=[('grpc.max_send_message_length', 512 * 1024 * 1024)
                                                    ,('grpc.max_receive_message_length', 512 * 1024 * 1024)])
        self.stub = compservice_pb2_grpc.compserviceStub(self.channel)
    def downloadModel(self, model):
        """
        Sends the server a request to download a HuggingFace model given by "model"

        Args:
            model(str): modelname
        
        Returns:
            boolean
        """
        return self.stub.downloadModel(compservice_pb2.Model(modelname=model)) 
    def deleteModel(self, model):
        """
        Sends a deletion request to the server
        
        Args:
            model(str): modelname
        Rerturns:
            boolean
        """
        return self.stub.deleteModel(compservice_pb2.Model(modelname=model)) 
    def getModels(self):
        """
        Sends a request to get the downloaded model from the server.

        Returns:
            Dictionary
        """
        _response = self.stub.getModels(compservice_pb2.Empty(empty=0)).model
        _response = {i.name : {'layers':i.layers, 'size':i.size} for i in _response}
        return _response

    def downloadStatic(self,model):
        """
        Sends the server a request to download a Gensim static model given by "model"

        Args:
            model(str): modelname
        
        Returns:
            boolean
        """
        return self.stub.downloadStatic(compservice_pb2.Model(modelname=model)) 
    
    def deleteStatic(self, model):
        """
        Sends a deletion request to the server
        
        Args:
            model(str): modelname
        Rerturns:
            boolean
        """
        return self.stub.deleteStatic(compservice_pb2.Model(modelname=model))
    
    def getStaticModels(self):
        """
        Sends a request to get the downloaded static models from the server.

        Returns:
            Dictionary
        """
        _response = self.stub.getStaticModels(compservice_pb2.Empty(empty=0)).model
        _response = {i.name : {'name':i.name} for i in _response}
        return _response

    def inf_session(self, _record, tab_record):
        """
        Sends an inference request to the server.

        Args:
            _record(dict): dictionary with the session inference requests.
            tab_record(dict): dictionary containing the visualisation tab for the results.
        Returns:
            Request
        """
        session_pb = compservice_pb2.Session()
        _request_list = []

        for i in list(_record.keys()):
            tab_record[i] = _record.pop(i)                      # Take request from request record
       

            request = compservice_pb2.Request()
            request.model = tab_record[i].model                 # Model to grpc message
            request.layer_low= tab_record[i].layer_low          # Lower layer to grpc message
            request.layer_up = tab_record[i].layer_up           # Upper layer to grpc message
            request.comp_func = tab_record[i].comp_func         # Composition Function to grpc message
            request.batchsize = tab_record[i].batchsize         # Batchsize to grpc message
            request.devices.name.extend(tab_record[i].devices)  # Device list to grpc message
            
            _dataset = tab_record[i].get_sentence_list()
            request.sentence.extend(_dataset)                   # Dataset sentences to grpc message
        
            _request_list.append(request)
        
        session_pb.request.extend(_request_list)
        
        for request in self.stub.inf_session(session_pb):
            yield request
    def getDevices(self):
        """
        Probes the server for the list of computing devices
        """
        device_list = self.stub.getDevices(compservice_pb2.Empty(empty=0))
        for device in device_list.dev:
            yield device

