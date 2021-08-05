import grpc
from src.grpc_files import compservice_pb2, compservice_pb2_grpc

class Server_grpc_if():
    def __init__(self, address):
        self.channel = grpc.insecure_channel(address)
        self.stub = compservice_pb2_grpc.compserviceStub(self.channel)
    def getDevices(self):
        yield self.stub.getDevices(compservice_pb2.Empty(empty=0))