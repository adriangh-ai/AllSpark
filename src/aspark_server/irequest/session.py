import multiprocessing as mp

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
    Transformer models
    """
    def __init__(self,model_type, modelname, layers, compfunc, partialdata, batchsize:int, single_dev, **kwargs):
        self.model_type = model_type
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

        Args: 
            output: Tuple
        """
        return output[2][self.lower_layer:self.upper_layer+1]

    def worker_start(self):
        """
        Instanciates the different parts and starts tokenization, inference and composition
         """
        time1 = time.time()
        embeddings = list()
        try:
            with torch.inference_mode():
                model = Model_factory.get_model(self.modelname, self.single_dev)
                composition = Comp_factory.get_compfun(self.compfunc)
                time1 = time.time()
                self.batchsize = max(model.paddding()*self.batchsize,1)
                batches = self._batching()
                # Inference loop
                for batch in batches:
                    tokens = model.tokenize(batch)
                    output = self._isolate_layers(model.inference(tokens))
                    output = composition.clean_special(output, model.special_mask)
                    output = composition.compose(output)
                    embeddings.append(output.cpu())

            # Concatenate results    
            embeddings = [ptensor.detach().numpy() for ptensor in embeddings]

            embeddings = numpy.concatenate(embeddings)
            pd_embeddings = pd.DataFrame(embeddings)

            # Post-processing
            cat_embeddings = pd.concat([self.partialdata, pd_embeddings], axis=1)
            cat_embeddings = cat_embeddings.sort_values(by='indx', ignore_index=True)
            cat_embeddings = cat_embeddings.drop(columns='indx')
        
        except Exception as e:
            print(f'Encountered an error during processing. {e}')
            raise  
        
        finally:
            #Memory cleanup
            model.to('cpu')
            del model
            del tokens
            del embeddings
            del output
            del composition
            del batches
            torch.cuda.empty_cache()
            gc.collect()
            
        time2 = time.time()
        print(f"tiempoes {time2-time1}")
        return cat_embeddings

class Req_worker_static():
    """
    Worker for static representation models.
    """
    def __init__(self,model_type, modelname, layers, compfunc, partialdata, batchsize:int, single_dev, **kwargs):
        self.model_type = model_type
        self.modelname = modelname
        self.lower_layer = 0
        self.upper_layer = 0
        self.compfunc = compfunc
        self.partialdata = partialdata
        self.batchsize = batchsize
        self.single_dev=single_dev
    def worker_start(self):
        time1 = time.time()
        try:
            model = Model_factory.get_model(self.modelname, self.single_dev)
            composition = Comp_factory.get_compfun(self.compfunc)
            time1 = time.time()
            tokens = model.tokenize(self.partialdata['sentence'].to_list())
            output =  model.inference(tokens)
            output = composition.compose(output)

            output = [ptensor.detach().numpy() for ptensor in output]
            output = pd.DataFrame(output)
            output = pd.concat([self.partialdata, output], axis = 1)

        except Exception as e:
            print(f'Encountered an error during processing. {e}')
            raise

        time2 = time.time()
        print(f'tiempo es {time2-time1}')
        return output

class InferenceRequest():
    """
    Represents a request made on a single model. Splits the data according to devices and feeds
    them to the different worker threads it spawns (one per device). Collects the results.
    """
    def __init__(self,model_type, model, layerLow, layerUp, compFunc, sentence, batchsize=16, devices = "cpu", **kwargs):
        self.id = 0
        self.model_type = model_type
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

        Args:
            output - Pandas DataFrame
        """
        return pd.concat(output, ignore_index=True)

    def worker_fn(self, args):
        """
        Method to map inference to each partial dataset
        """
        #REDO without factory
        worker = None
        if 'transformer' in args['model_type']:
            worker = Req_worker(**args)
        else:
            worker = Req_worker_static(**args)
        return worker.worker_start()

    def inference_run(self): 
 
        """
        Creates the worker processes and stores them in a pool to later collect the results.
        It divides the dataset among devices, rounding up, takes the maximum between the resulting size
        and the batch size to establish a lower portion limit and spawns (as opposed to fork()) one worker per 
        portion per GPU.
        """
        if 'static' in self.model_type:
            self.devs = ['cpu']

        jump = max(math.ceil(len(self.req_dataset)/len(self.devs)) , self.batchsize)

        parcial_inf = []
        for i in range(0,len(self.req_dataset),jump):
            parcial_inf.append({'model_type': self.model_type
                                ,'modelname':self.modelname
                                ,'layers':self.layers
                                ,'compfunc':self.compfunc
                                ,'correction':self.correction
                                ,'partialdata':self.req_dataset[i:i+jump]
                                ,'batchsize':self.batchsize
                                ,'single_dev':self.devs[(i//jump)%len(self.devs)]})
 
        with ThreadPoolExecutor(max_workers=len(self.devs)) as ex:
            worker_ls = ex.map(self.worker_fn , parcial_inf)

            
        try:
            embeddings = self._join_ouputs(worker_ls)
        except Exception as e:
            print('Propagating Thread Error.')
            raise
        finally:
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
        """
        Initialises the Inference object and runs it.
        Return: DataFrame
        """
        inf = InferenceRequest(**args)
        try:
            output = inf.inference_run()
        except Exception as e:
            print('Error. Returning error value.')
            return pd.DataFrame({'sentence': 'error', 0:[1], 1:[1], 2:[1]})
        return output

    def session_run(self):
        """
        Spawns separate processes for each request.
        """
        inference_ls = []
        with ProcessPoolExecutor(mp_context=mp.get_context('spawn')) as ex:
            inference_ls = ex.map(self.inference_fn, self.reqparams)

        return [i for i in inference_ls]
            


if __name__ == "__main__":
    """
    Test case.
    """
    from distributed import Client
    client = Client()
    import os
    cur_dir = os.path.dirname(__file__)
    # Load a Dataset
    with open("/home/tardis/Documents/wikisent2.txt", "r", encoding="utf-8") as f:
        lines = [f.read()]
    lines = lines[0].splitlines()
    xdataset = lines[:6000]
    dasta = pd.DataFrame({'sentence': ['This is a test of a test.', "I have a dog."]})
    dasta2 = pd.DataFrame([{'sentence': 'Testing this.', 'position':0},{'sentence': 'This.', 'position':1}])
    dasta3 = pd.DataFrame(['This is a test.', "Test.", "Testing this."])
    dasta3.columns = ['sentence']
    ses = Session([{'model_type':'static',
                    'model':'fasttext-wiki-news-subwords-300', 
                    'layerLow': 12, 
                    'layerUp': 12, 
                    'compFunc': 'f_inf', 
                    'sentence': dasta3 , 
                    'batchsize': 16, 
                    'devices': {'name': ['cuda:1', 'cuda:2']}}])
    print(ses.session_run())
"""     lista = []
    with ThreadPoolExecutor() as ex:
        lista.append(ex.submit(ses.session_run))
        #lista.append(ex.submit(ses3.session_run))
        #concurrent.futures.wait(lista, return_when="FIRST_EXCEPTION")
        counter = 0
        while not counter == 15:
            counter+=1
            time.sleep(1)
            print(f"{counter}")
        concurrent.futures.wait(lista, return_when="FIRST_EXCEPTION")
    for i in lista:
        print(i.result())
    #print(ses.session_run()) """
