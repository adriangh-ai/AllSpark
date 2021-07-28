import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from collections import deque
import itertools
from threading import TIMEOUT_MAX

import torch

import math

import time

from models import model_cat, basemodel
from models.model_cat import Model_factory
from composition.composition_func import Comp_factory


class Req_worker():
    """
    Worker, one for each model to run in parallel, on each GPU provided, for data paralellism 
    """
    def __init__(self, modelname, layers, compfunc, partialdata, batchsize:int, single_dev, **kwargs):
        self.modelname = modelname
        self.lower_layer = layers[0]
        self.upper_layer = layers[1]
        self.compfunc = compfunc
        self.partialdata = partialdata
        self.batchsize = batchsize
        self.single_dev=single_dev
        
        allowed_keys = {'correction'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def _batching(self):
        """
        Process the input in batches. Smart batching?? preserve cumulative order of batches
        """
        #_sorted = list(itertools.chain(i['text'] for i in self.partialdata)) #substitute for tag
        _sorted = self.partialdata
        for a in range(0, len(_sorted), self.batchsize):
            yield _sorted[a:a+self.batchsize]
    
    def _isolate_layers(self, output):
        """
        Slices the output, selecting only the layers to be further processed
        """
        return output[2][self.lower_layer:self.upper_layer+1]

    def worker_start(self):
        """
        Instanciates the different parts and starts tokenization, inference and composition
         """
        #correction

        output = list()
        batches = self._batching()

        with torch.inference_mode():
            model = Model_factory.get_model(self.modelname, self.single_dev)
            composition = Comp_factory.get_compfun(self.compfunc)
            time1 = time.time()
            for batch in batches:
                tokens = model.tokenize(batch)
                output = self._isolate_layers(model.inference(tokens))

                output = composition.clean_special(output, model.special_mask)
                output = composition.compose(output)
            time2 = time.time()
            print(f"tiempoes {time2-time1}")

        return [i.cpu() for i in output]

class InferenceRequest():
    """
    Represents a request made to a single model. Splits the data according to devices and feeds
    them to the different worker processes it spawns (one per device). Collects the results.
    """
    def __init__(self, id, modelname, layers, compfunc, ses_dataset, batchsize=16, devs = "cpu", **kwargs):
        self.id = id
        self.devs = devs
        self.modelname=modelname
        self.compfunc=compfunc
        self.req_dataset=ses_dataset
        self.layers=layers
        self.batchsize = batchsize
        self.correction = ''

        allowed_keys = {'correction'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
    def _join_ouputs(self, output):
        """
        Joins the different batches of each iteration
        """
        pass

    def inference_run(self): 
 
        """
        Creates the worker processes and stores them in a pool to later collect the results.
        It divides the dataset among devices, rounding up, takes the maximum between the resulting size
        and the batch size to establish a lower portion limit and spawns (as opposed to fork()) one worker per 
        portion per GPU.
        """
        with ThreadPoolExecutor(max_workers=len(self.devs)) as ex:
            workerqueue = list()
            jump = max(math.ceil(len(self.req_dataset)/len(self.devs)) , self.batchsize) #Portion per GPU, min size batchsize
            
            for i in range(0,len(self.req_dataset),jump):
                workerqueue.append(ex.submit(Req_worker(modelname=self.modelname
                                                    ,layers=self.layers
                                                    ,compfunc=self.compfunc
                                                    ,correction=self.correction
                                                    ,partialdata=self.req_dataset[i:i+jump]
                                                    ,batchsize=self.batchsize
                                                    ,single_dev=self.devs[(i//jump)%len(self.devs)]).worker_start))
            
            embeddings = [i.result() for i in workerqueue]
        
        return embeddings
            

class Session():
    """
    Represents a conglomerate of requests from a single user, made of different models to different devices.
    """
    def __init__(self, id, *args):
        self.id = id
        self.reqparams = args

    def session_run(self):
        with ThreadPoolExecutor() as ex:
            output_rq = list()
            for i in self.reqparams:
                request_i = InferenceRequest(*i)
                output_rq.append(ex.submit(request_i.inference_run))
            output_embeddings= concurrent.futures.wait(output_rq, return_when="FIRST_EXCEPTION")
        output_embeddings = list(i.result() for i in output_embeddings.done)
        return output_embeddings
            


if __name__ == "__main__":
    """
    For testing
    """
    import os
    cur_dir = os.path.dirname(__file__)
    with open("/home/tardis/Documents/wikisent2.txt", "r", encoding="utf-8") as f:
        lines = [f.read()]
    lines = lines[0].splitlines()
    xdataset = lines[:1000]
    
    ses = Session(12, [123, 'bert-base-uncased', [12,12], 'cls', xdataset , 16, ['cuda:1','cuda:2']]
                        ,[123, 'bert-base-uncased', [12,12], 'cls', xdataset , 16, ['cuda:2']])
    #ses3 = Session(12, [123, 'bert-base-uncased', [11,12], 'cls', xdataset , 2, ['cuda:1','cuda:2']]
    #                    ,[123, 'bert-base-uncased', [11,12], 'cls', xdataset , 16, ['cuda:2']])
    ses3 = Session(12, [123, 'bert-base-uncased', [12,12], 'f_inf', xdataset , 16, ["cuda:1", "cuda:2"]])
    #ses = Session(12, [123, 'bert-base-uncased', [12,12], 'f_joint', xdataset[1500:] , 32, ["cuda:2"]])
    lista = []
    with ThreadPoolExecutor() as ex:
        lista.append(ex.submit(ses.session_run))
        lista.append(ex.submit(ses3.session_run))
        #concurrent.futures.wait(lista, return_when="FIRST_EXCEPTION")
        print("we still going")
        
        """ counter = 0
        while not counter == 15:
            counter+=1
            time.sleep(1)
            print(f"{counter}") """
        concurrent.futures.wait(lista, return_when="FIRST_EXCEPTION")
    for i in lista:
        print(i.result())
    #print(ses.session_run())
