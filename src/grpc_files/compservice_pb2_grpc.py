# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import compservice_pb2 as compservice__pb2


class compserviceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.downloadModel = channel.unary_unary(
                '/compservice/downloadModel',
                request_serializer=compservice__pb2.Model.SerializeToString,
                response_deserializer=compservice__pb2.Response.FromString,
                )
        self.deleteModel = channel.unary_unary(
                '/compservice/deleteModel',
                request_serializer=compservice__pb2.Model.SerializeToString,
                response_deserializer=compservice__pb2.Response.FromString,
                )
        self.getModels = channel.unary_unary(
                '/compservice/getModels',
                request_serializer=compservice__pb2.Empty.SerializeToString,
                response_deserializer=compservice__pb2.ModelList.FromString,
                )
        self.updloadDataset = channel.stream_unary(
                '/compservice/updloadDataset',
                request_serializer=compservice__pb2.Dataset.SerializeToString,
                response_deserializer=compservice__pb2.Response.FromString,
                )
        self.deleteDataset = channel.unary_unary(
                '/compservice/deleteDataset',
                request_serializer=compservice__pb2.DatasetName.SerializeToString,
                response_deserializer=compservice__pb2.Response.FromString,
                )
        self.getDatasets = channel.unary_unary(
                '/compservice/getDatasets',
                request_serializer=compservice__pb2.Empty.SerializeToString,
                response_deserializer=compservice__pb2.DatasetList.FromString,
                )
        self.inf_session = channel.unary_stream(
                '/compservice/inf_session',
                request_serializer=compservice__pb2.Session.SerializeToString,
                response_deserializer=compservice__pb2.EmbeddingData.FromString,
                )
        self.getDevices = channel.unary_unary(
                '/compservice/getDevices',
                request_serializer=compservice__pb2.Empty.SerializeToString,
                response_deserializer=compservice__pb2.DeviceList.FromString,
                )


class compserviceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def downloadModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def deleteModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getModels(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def updloadDataset(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def deleteDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getDatasets(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def inf_session(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def getDevices(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_compserviceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'downloadModel': grpc.unary_unary_rpc_method_handler(
                    servicer.downloadModel,
                    request_deserializer=compservice__pb2.Model.FromString,
                    response_serializer=compservice__pb2.Response.SerializeToString,
            ),
            'deleteModel': grpc.unary_unary_rpc_method_handler(
                    servicer.deleteModel,
                    request_deserializer=compservice__pb2.Model.FromString,
                    response_serializer=compservice__pb2.Response.SerializeToString,
            ),
            'getModels': grpc.unary_unary_rpc_method_handler(
                    servicer.getModels,
                    request_deserializer=compservice__pb2.Empty.FromString,
                    response_serializer=compservice__pb2.ModelList.SerializeToString,
            ),
            'updloadDataset': grpc.stream_unary_rpc_method_handler(
                    servicer.updloadDataset,
                    request_deserializer=compservice__pb2.Dataset.FromString,
                    response_serializer=compservice__pb2.Response.SerializeToString,
            ),
            'deleteDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.deleteDataset,
                    request_deserializer=compservice__pb2.DatasetName.FromString,
                    response_serializer=compservice__pb2.Response.SerializeToString,
            ),
            'getDatasets': grpc.unary_unary_rpc_method_handler(
                    servicer.getDatasets,
                    request_deserializer=compservice__pb2.Empty.FromString,
                    response_serializer=compservice__pb2.DatasetList.SerializeToString,
            ),
            'inf_session': grpc.unary_stream_rpc_method_handler(
                    servicer.inf_session,
                    request_deserializer=compservice__pb2.Session.FromString,
                    response_serializer=compservice__pb2.EmbeddingData.SerializeToString,
            ),
            'getDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.getDevices,
                    request_deserializer=compservice__pb2.Empty.FromString,
                    response_serializer=compservice__pb2.DeviceList.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'compservice', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class compservice(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def downloadModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/compservice/downloadModel',
            compservice__pb2.Model.SerializeToString,
            compservice__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def deleteModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/compservice/deleteModel',
            compservice__pb2.Model.SerializeToString,
            compservice__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getModels(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/compservice/getModels',
            compservice__pb2.Empty.SerializeToString,
            compservice__pb2.ModelList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def updloadDataset(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_unary(request_iterator, target, '/compservice/updloadDataset',
            compservice__pb2.Dataset.SerializeToString,
            compservice__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def deleteDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/compservice/deleteDataset',
            compservice__pb2.DatasetName.SerializeToString,
            compservice__pb2.Response.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getDatasets(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/compservice/getDatasets',
            compservice__pb2.Empty.SerializeToString,
            compservice__pb2.DatasetList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def inf_session(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_stream(request, target, '/compservice/inf_session',
            compservice__pb2.Session.SerializeToString,
            compservice__pb2.EmbeddingData.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def getDevices(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/compservice/getDevices',
            compservice__pb2.Empty.SerializeToString,
            compservice__pb2.DeviceList.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
