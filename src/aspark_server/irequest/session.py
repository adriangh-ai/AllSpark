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


class preprocess_worker():
    """
    Worker, one for each model to run in parallel, on each GPU provided, for data paralellism 
    """
    def __init__(self, modelname, layers, compfunc, partialdata, batchsize:int, single_dev, **kwargs):
        self.modelname = modelname
        self.lower_layer = layers[0]
        self.upper_layer = layers[1]
        self.compfun = compfunc
        self.partialdata = partialdata
        self.batchsize = batchsize
        self.single_dev=single_dev
        #self.compfunc = factory(compfunc)
        
        allowed_keys = {'correction'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def _batching(self):
        """
        Process the input in batches. Smart batching?? preserve cumulative order of batches
        """
        #_sorted = list(itertools.chain(i['text'] for i in self.partialdata)) #substitute for tag
        _batchsize = int(len(self.devs)*self.batchsize)
        _sorted = self.req_dataset
        for a in range(0, len(_sorted), _batchsize):
            yield _sorted[a:a+_batchsize]

    def worker_start(self):
        """
        Instanciates the different parts and starts tokenization, inference and composition
         """
        model = Model_factory.get_model(self.modelname).to(self.single_dev)
        composition = Comp_factory.get_compfun(self.compfun)
        
        #correction
        output = []
        batches = self._batching()

        try:
            with torch.no_grad():
                #for a in range(0, len(self.partialdata), self.batchsize):
                    #tokens = self.model.tokenize(self.partialdata[a:a+self.batchsize])
                for batch in batches:
                    tokens = model.tokenize(batch)
                    if "cuda" in self.single_dev:
                        tokens = tokens.to(self.single_dev) #test performance
                    output = list(self._isolate_layers(model.inference(tokens)))
                    output = composition.clean_special(output, model.special_ids, tokens["input_ids"]) 
                    
                    #self.output = composition.clean_special(self.model.all_special_ids(), tokens["input_ids"])
                    #torch.stack
        except Exception as e: #temporal until out of memory exception
            print(e)
        #self.output = composition.compose(self.output)
        #output = recomponer tags y tensors
        output = [i.cpu() for i in output]
        return output
class composition_worker():
    def __init__(self):
        pass
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
        self.lower_layer = layers[0]
        self.upper_layer = layers[1]
        self.batchsize = batchsize
        self.correction = ''

        allowed_keys = {'correction'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def _batching(self):
        """
        Process the input in batches. Smart batching?? preserve cumulative order of batches
        """
        #_sorted = list(itertools.chain(i['text'] for i in self.partialdata)) #substitute for tag
        _batchsize = int(len(self.devs)*self.batchsize)
        _sorted = self.req_dataset
        for a in range(0, len(_sorted), _batchsize):
            yield _sorted[a:a+_batchsize]

    def _isolate_layers(self, output):
        """
        Slices the output, selecting only the layers to be further processed
        """
        return output[2][self.lower_layer:self.upper_layer+1]
        
    def _join_ouputs(self, output):
        """
        Joins the different batches of each iteration
        """
        pass

    def inference_run(self): 
        output = list()
        batches = self._batching()

        with torch.inference_mode():
            model = Model_factory.get_model(self.modelname, self.devs)
            composition = Comp_factory.get_compfun(self.compfunc)
            time1 = time.time()
            for batch in batches:
                tokens = model.tokenize(batch)
                output = self._isolate_layers(model.inference(tokens))

                #output = composition.clean_special(output, model.special_mask)
                #output = composition.compose(output)
            time2 = time.time()
            print(f"tiempoes {time2-time1}")

        return "done" #[i.cpu() for i in output]
            

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
    xdataset = lines[:3000]
    
    #ses = Session(12, [123, 'bert-base-uncased', [11,12], 'cls', xdataset , 16, ['cuda:1','cuda:2']]
                        #,[123, 'bert-base-uncased', [11,12], 'cls', xdataset , 16, ['cuda:2']])
    #ses3 = Session(12, [123, 'bert-base-uncased', [11,12], 'cls', xdataset , 2, ['cuda:1','cuda:2']]
    #                    ,[123, 'bert-base-uncased', [11,12], 'cls', xdataset , 16, ['cuda:2']])
    ses3 = Session(12, [123, 'bert-base-uncased', [12,12], 'f_joint', xdataset[:1500] , 32, ["cuda:1"]])
    ses = Session(12, [123, 'bert-base-uncased', [12,12], 'f_joint', xdataset[1500:] , 32, ["cuda:2"]])
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
