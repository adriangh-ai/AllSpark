import torch
import transformers as ts
import os, random, shutil, psutil
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import grpc
from src.grpc_files import compservice_pb2_grpc, compservice_pb2

from irequest import devices, session
from irequest.composition import *


WORKDIR = Path(__file__).parent             #Program base file tree
MODEL_LIST_JSON = Path.joinpath(WORKDIR, f"irequest/models/stored_models.json") 

### SERVER INTERFACE

class CompServiceServicer(compservice_pb2_grpc.compserviceServicer):
    """ 
    Methods for the gRPC server to provide the service
    """
    def __init__(self):
        """
        Computing devices:
        If cuda is available, fills the gpus, then adds cpu.
        """
        self.ls_dev = [devices.Dev(  name = torch.cuda.get_device_name(i)
                                    ,id = f"cuda:{i}"
                                    ,device_type = "gpu") for i in 
                                        range(torch.cuda.device_count()) if torch.cuda.is_available()]

        self.ls_dev.append(devices.Dev(  name = 'cpu'
                                    ,id = 'cpu'
                                    ,n_cores = os.cpu_count()
                                    ,device_type = "cpu"))   #add cpu to the device list
        #Persistent Model list JSON storage
        self.model_list = self._get_models_from_file()       #Loads model list from file
        self._models_downloading = {}                        #Download temporary record
    
    def _get_models_from_file(self):
        data = {}
        if MODEL_LIST_JSON.exists():
            with open(MODEL_LIST_JSON, 'r') as data_file:
                data = json.load(data_file)

        return data
            
    def downloadModel(self, request, context):
        """
        Creates a thread that downloads the model, config file and tokenizer storing all in disk.
        Deletes huggingface cache.
        Args: 
            model : str - model name, acording to Huggingface's respository.
        """
        model = request.modelname
        if model in self._models_downloading:                                  #GUARD
            if 'downloading' in self._models_downloading[model]:
                _lock = self._models_downloading[model]['downloading']
                try:
                    _lock.acquire()
                finally:
                    _lock.release()                       
                    return compservice_pb2.Response(completed=True)

        if not model in self.model_list:            
            lock = threading.Lock()
            self._models_downloading[model]={'downloading':lock}
            
            #Local model cache
            model_cache = Path.joinpath(WORKDIR, f"irequest/models/cache/{model}{random.randint(1000,9999)}")   
            model_folder = Path.joinpath(WORKDIR, f"irequest/models/{model}")
            try: 
                lock.acquire()
                
                Path(model_folder).mkdir(parents=True, exist_ok=True)
                tmodel = ts.AutoModel.from_pretrained(model, cache_dir=model_cache)  
                tmodel.save_pretrained(model_folder)                         #Save Model
                _tmodel_conf = tmodel.config.to_diff_dict()
                _tmodel_conf['num_hidden_layers'] = tmodel.config.num_hidden_layers
                _tmodel_conf['num_param'] = tmodel.num_parameters()          #Add number of parameters to info
                self.model_list[model]= _tmodel_conf
                #lock?
                with open(MODEL_LIST_JSON, 'w+') as stored_models:
                    json.dump(self.model_list, stored_models)

                ttokenizer = ts.AutoTokenizer.from_pretrained(model, cache_dir=model_cache) 
                ttokenizer.save_pretrained(model_folder)                     #Save Tokenizer  
            except FileExistsError as e:
                print(e)
            finally:
                lock.release()
                self._models_downloading.pop(model)

            if model_cache.exists():
                try:
                    shutil.rmtree(model_cache)                               #Remove cache folder
                except FileNotFoundError as e:
                    print(f"Could not remove {model_cache}: File not found.")

        return compservice_pb2.Response(completed=True)

    def deleteModel(self, request, context):
        model = request.modelname
        try:
            if model in self.model_list:
                self.model_list.pop(model)
                 #lock?
                with open(MODEL_LIST_JSON, 'w+') as stored_models:
                    json.dump(self.model_list, stored_models)
        
            shutil.rmtree(Path.joinpath(WORKDIR, f"irequest/models/{model}"))   #Remove model folder
        except FileNotFoundError as e:
            print(e)
            print(f"Could not remove {model}: File not found.")
            return compservice_pb2.Response(completed=False)
        
        return compservice_pb2.Response(completed=True)
        
        
    def getModels(self, request, context):
        models_response = compservice_pb2.ModelList()
        _model_list = []
        for modelcfg in self.model_list.values():
            _model_list.append(compservice_pb2.ModelStruct(name=modelcfg['_name_or_path'],
                                                        layers=modelcfg['num_hidden_layers'],
                                                        size = modelcfg['num_param']))
        
        models_response.model.extend(_model_list)
        return models_response

    def upl_dataset():
        pass

    def inf_session(self, request, context):
        print(request.request[0].model)
        return super().inf_session(request, context)

    def getDevices(self, request, context):
        devices = compservice_pb2.DeviceList()
        _ls_dev = []
        for _device in self.ls_dev:
            _ls_dev.append(devices.Device(device_name=_device.name, 
                                            id= _device.id,
                                            memory_total=_device.memory, 
                                            memory_free=_device.mem_update()))

        devices.dev.extend(_ls_dev)
        return devices

def serve():
    """
    Method to start the grpc server
    """
    server= grpc.server(ThreadPoolExecutor(max_workers=20))
    compservice_pb2_grpc.add_compserviceServicer_to_server(CompServiceServicer(), server)
    server.add_insecure_port('[::]:42001') #add local from params? add secure?
    server.start()
    server.wait_for_termination()
    server.stop()


if __name__ == '__main__':

    print("Starting server")
    serve()
       
    

#Composition ops

#Anisotropy

#Saved models

#Saved datasets

#Password settings

