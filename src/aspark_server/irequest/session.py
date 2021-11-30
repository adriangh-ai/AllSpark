import multiprocessing as mp
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import gc

import pandas as pd
pd.options.mode.chained_assignment = None

import torch
import numpy

import math

import time

from .models.model_cat import Model_factory
from .composition.composition_func import Comp_factory

""" from models import model_cat, basemodel
from models.model_cat import Model_factory
from composition.composition_func import Comp_factory """

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
        Process the input in batches. Performs Smart Batching sorting the input and tokenizing it,
        then dividing it into chunks of size bactchsize
        """
        self.partialdata['indx'] = self.partialdata.reset_index().index
        print(self.partialdata)
        self.partialdata = self.partialdata.sort_values(by='sentence'
                                                        ,ignore_index=True
                                                        ,key=lambda x:x.str.split().map(len))
        _sorted = self.partialdata['sentence'].to_list()
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
        time1 = time.time()
        embeddings = list()
        try:
            with torch.inference_mode():
                model = Model_factory.get_model(self.modelname, self.single_dev)
                composition = Comp_factory.get_compfun(self.compfunc)
                time1 = time.time()
                self.batchsize = max(model.paddding()*self.batchsize,1)
                batches = self._batching()
                for batch in batches:
                    tokens = model.tokenize(batch)
                    output = self._isolate_layers(model.inference(tokens))
                    output = composition.clean_special(output, model.special_mask)
                    output = composition.compose(output)
                    embeddings.append(output.cpu())
                
            embeddings = [ptensor.detach().numpy() for ptensor in embeddings]

            embeddings = numpy.concatenate(embeddings)
            pd_embeddings = pd.DataFrame(embeddings)

            model.to('cpu')
            #Memory cleanup
            del model
            del tokens
            del embeddings
            del output
            del composition
            del batches
            torch.cuda.empty_cache()
            gc.collect()

            cat_embeddings = pd.concat([self.partialdata, pd_embeddings], axis=1)
            cat_embeddings = cat_embeddings.sort_values(by='indx', ignore_index=True)
            cat_embeddings = cat_embeddings.drop(columns='indx')
        except Exception as e:
            print(f'Encountered and error during processing. {e}')
            
        time2 = time.time()
        print(f"tiempoes {time2-time1}")
        return cat_embeddings

class InferenceRequest():
    """
    Represents a request made to a single model. Splits the data according to devices and feeds
    them to the different worker processes it spawns (one per device). Collects the results.
    """
    def __init__(self, model, layerLow, layerUp, compFunc, sentence, batchsize=16, devices = "cpu", **kwargs):
        self.id = 0
        self.devs = devices['name']
        self.modelname=model
        self.compfunc=compFunc
        self.req_dataset=sentence
        self.layers=[layerLow, layerUp]
        self.batchsize = batchsize
        self.correction = ''

        allowed_keys = {'correction'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)
        
    def _join_ouputs(self, output):
        """
        Joins the different batches of each iteration
        """
        return pd.concat(output, ignore_index=True)

    def worker_fn(self, args):
        worker = Req_worker(**args)
        return worker.worker_start()

    def inference_run(self): 
 
        """
        Creates the worker processes and stores them in a pool to later collect the results.
        It divides the dataset among devices, rounding up, takes the maximum between the resulting size
        and the batch size to establish a lower portion limit and spawns (as opposed to fork()) one worker per 
        portion per GPU.
        """
        jump = max(math.ceil(len(self.req_dataset)/len(self.devs)) , self.batchsize)

        parcial_inf = []
        for i in range(0,len(self.req_dataset),jump):
            parcial_inf.append({'modelname':self.modelname
                                ,'layers':self.layers
                                ,'compfunc':self.compfunc
                                ,'correction':self.correction
                                ,'partialdata':self.req_dataset[i:i+jump]
                                ,'batchsize':self.batchsize
                                ,'single_dev':self.devs[(i//jump)%len(self.devs)]})

        with ThreadPoolExecutor(max_workers=len(self.devs)) as ex:
            worker_ls = ex.map(self.worker_fn , parcial_inf)
        
        embeddings = self._join_ouputs(worker_ls)
        
        del self.req_dataset
        gc.collect()

        return embeddings
            

class Session():
    """
    Represents a conglomerate of requests from a single user, made of different models to different devices.
    """
    def __init__(self, args):
        self.id = 0
        self.reqparams = args
    
    def inference_fn(self, args):
        inf = InferenceRequest(**args)
        return inf.inference_run()

    def session_run(self):

        inference_ls = []
        with ProcessPoolExecutor(mp_context=mp.get_context('spawn')) as ex:
            inference_ls = ex.map(self.inference_fn, self.reqparams)

        return [i for i in inference_ls]
            


if __name__ == "__main__":
    """
    For testing
    """
    from distributed import Client
    client = Client()
    import os
    cur_dir = os.path.dirname(__file__)
    with open("/home/tardis/Documents/wikisent2.txt", "r", encoding="utf-8") as f:
        lines = [f.read()]
    lines = lines[0].splitlines()
    xdataset = lines[:6000]
    """ xdatasett = pd.DataFrame(xdataset)
    xdatasett['sentence']= xdataset """
    
    #ses = Session(12, [123, 'bert-base-uncased', [12,12], 'cls', xdataset , 16, ['cuda:1','cuda:2']]
     #                   ,[123, 'bert-base-uncased', [12,12], 'cls', xdataset , 16, ['cuda:2']])
    #ses3 = Session(12, [123, 'bert-base-uncased', [11,12], 'cls', xdataset , 2, ['cuda:1','cuda:2']]
                       # ,[123, 'bert-base-uncased', [11,12], 'cls', xdataset , 16, ['cuda:2']])
    #ses = Session(12, [123, 'bert-base-uncased', [11,12], 'f_inf', xdataset , 16, ["cuda:1", "cuda:2"]])
    #ses = Session(1,[{'id':123, 'modelname':'bert-base-uncased', 'layers':[11,12], 'compfunc':'f_inf', 'ses_dataset':xdataset, 'batchsize':16, 'devs':["cuda:1",'cuda:2']}])
    dasta = pd.DataFrame({'sentence': ['This is a test of a test.', "I'm not really sure what the fuck I'm doing."]})
    dasta2 = pd.DataFrame([{'sentence': 'Testing this.', 'position':0},{'sentence': 'This.', 'position':1}])
    dasta3 = pd.DataFrame(['This is a test of a test.', "I'm not really sure what the fuck I'm doing."])
    dasta3.columns = ['sentence']
    ses = Session([{'model': '/home/tardis/Documents/uned/PFG/AllSpark/src/aspark_server/irequest/models/bert-base-uncased', 'layerLow': 12, 'layerUp': 12, 'compFunc': 'cls', 'sentence': dasta3 , 'batchsize': 16, 'devices': {'name': ['cuda:1', 'cuda:2']}}])
                    #{'model': 'bert-base-uncased', 'layerLow': 12, 'layerUp': 12, 'compFunc': 'cls', 'data': xdatasett , 'batchsize': 16, 'devices': {'name': ['cuda:2']}}])
    #ses = Session(12, [123, 'bert-base-uncased', [12,12], 'f_ind', ["Testing this.","this."] , 6, ["cuda:1","cuda:2"]])
    print(ses.session_run())
"""     lista = []
    with ThreadPoolExecutor() as ex:
        lista.append(ex.submit(ses.session_run))
        #lista.append(ex.submit(ses3.session_run))
        #concurrent.futures.wait(lista, return_when="FIRST_EXCEPTION")
        print("we still going")
        
        counter = 0
        while not counter == 15:
            counter+=1
            time.sleep(1)
            print(f"{counter}")
        concurrent.futures.wait(lista, return_when="FIRST_EXCEPTION")
    for i in lista:
        print(i.result())
    #print(ses.session_run()) """
