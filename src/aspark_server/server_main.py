import torch
import transformers as ts
import asyncio
import os, random, shutil, psutil
import threading
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import grpc
from src.grpc_files import compservice_pb2_grpc, compservice_pb2

from irequest import devices, session
from irequest.composition import *


WORKDIR = Path(__file__).parent             #Program base file tree
### SERVER INTERFACE
def downloadModel(model:str, lock:threading.Lock):
    """
    Creates a thread that ownloads the model, config file and tokenizer storing all in disk.
    Deletes huggingface cache.
    Args: 
        model : str - model name, acording to Huggingface's respository.
    """
    def _download(model):
        model_cache = Path.joinpath(WORKDIR, f"irequest/models/cache/{model}{random.randint(1000,9999)}")    #Local model cache
        model_folder = Path.joinpath(WORKDIR, f"irequest/models/{model}")
        Path(model_folder).mkdir(parents=True, exist_ok=True)
 
        model_file = Path(Path.joinpath(model_folder, "pytorch_model.bin" ))
        if not model_file.exists():    
            tmodel = ts.AutoModel.from_pretrained(model, cache_dir=model_cache)  
            lock.acquire()
            try:
                tmodel.save_pretrained(model_folder)        #Save Model
            except FileExistsError as e:
                print(e)
            lock.release()

        tokenizer_file = Path(Path.joinpath(model_folder, "tokenizer.json"))       
        if not tokenizer_file.exists():
            ttokenizer = ts.AutoTokenizer.from_pretrained(model, cache_dir=model_cache) 
            lock.acquire()
            try:
                ttokenizer.save_pretrained(model_folder)    #Save Tokenizer  
            except FileExistsError as e:
                print(e)
            lock.release()
         
        lock.acquire()
        try:
            shutil.rmtree(model_cache)                      #Remove cache folder
        except FileNotFoundError as e:
            print(f"Could not remove {model_cache}: File not found.")
        lock.release()
    t = threading.Thread(target=_download(model), daemon=True)

def rm_model(model:str):
    pass

def upl_dataset():
    pass

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

    
    def getDevices(self, request, context):
        return compservice_pb2.DeviceList(device_name=(i.name for i in self.ls_dev))

def serve():
    """
    Method to start the grpc server
    """
    server= grpc.server(ThreadPoolExecutor(max_workers=10))
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

