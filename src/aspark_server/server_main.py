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
### SERVER INTERFACE


###Initialisation scripts
#Folders and config files





class CompServiceServicer(compservice_pb2_grpc.compserviceServicer):
    """ 
    Methods for the gRPC server to provide the service
    """
    def __init__(self):
        """
        Computing devices:
        If cuda is available, fills the gpus, then adds cpu.
        """
        self.ls_dev = [devices.Dev(  name = f"cuda:{i}" 
                                ,device_type = "gpu") for i in 
                                        range(torch.cuda.device_count()) if torch.cuda.is_available()]

        self.ls_dev.append(devices.Dev(  name = 'cpu'
                                    ,n_cores = os.cpu_count()
                                    ,device_type = "cpu"))   #add cpu to the device list
        self.model_list = self._get_models_from_file()
    
    def _get_models_from_file(self):
        data = {}
        if Path('stored_models.json').exists():
            with open('stored_models.json', 'r') as data_file:
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
        model_cache = Path.joinpath(WORKDIR, f"irequest/models/cache/{model}{random.randint(1000,9999)}")    #Local model cache
        model_folder = Path.joinpath(WORKDIR, f"irequest/models/{model}")
        Path(model_folder).mkdir(parents=True, exist_ok=True)

        model_file = Path(Path.joinpath(model_folder, "pytorch_model.bin" ))
        if not model_file.exists():    
            tmodel = ts.AutoModel.from_pretrained(model, cache_dir=model_cache)  
            try:
                tmodel.save_pretrained(model_folder)        #Save Model
            except FileExistsError as e:
                print(e)
                return compservice_pb2.Response(completed=False)
            
        tokenizer_file = Path(Path.joinpath(model_folder, "tokenizer.json"))       
        if not tokenizer_file.exists():
            ttokenizer = ts.AutoTokenizer.from_pretrained(model, cache_dir=model_cache) 
            try:
                ttokenizer.save_pretrained(model_folder)    #Save Tokenizer  
            except FileExistsError as e:
                print(e)
                return compservice_pb2.Response(completed=False)
        try:
            shutil.rmtree(model_cache)                      #Remove cache folder
        except FileNotFoundError as e:
            print(f"Could not remove {model_cache}: File not found.")

        self.model_list[model]= tmodel.config.to_diff_dict()
        with open(Path.joinpath(WORKDIR, f"irequest/models/stored_models.json"), 'w+') as stored_models:
            json.dump(self.model_list, stored_models)

        return compservice_pb2.Response(completed=True)

    def deleteModel(self, request, context):
        model = request.modelname
        try:
            shutil.rmtree(Path.joinpath(WORKDIR, f"irequest/models/{model}"))   #Remove model folder
        except FileNotFoundError as e:
            print(e)
            print(f"Could not remove {model}: File not found.")
            return compservice_pb2.Response(completed=False)
        self.model_list.pop(model)
        with open(Path.joinpath(WORKDIR, f"irequest/models/stored_models.json"), 'w+') as stored_models:
            json.dump(self.model_list, stored_models)
        return compservice_pb2.Response(completed=True)
        

    def upl_dataset():
        pass

    
    def getDevices(self, request, context):
        devices = compservice_pb2.DeviceList()
        devices.device_name.extend([i.name for i in self.ls_dev])
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

