import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch
import itertools
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
        _sorted = list(itertools.chain(i['text'] for i in self.partialdata))

        for a in range(0, len(_sorted), self.batchsize):
            yield _sorted[a:a+self.batchsize]

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
                    output.append(composition.clean_special(output, model.special_ids, tokens["input_ids"]))  
                    
                    #self.output = composition.clean_special(self.model.all_special_ids(), tokens["input_ids"])
                    #torch.stack
        except Exception as e: #temporal until out of memory exception
            print(e)
        #self.output = composition.compose(self.output)
        #output = recomponer tags y tensors
        return output

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
        self.ses_dataset=ses_dataset
        self.layers = layers
        self.batchsize = batchsize
        self.correction = ''

        allowed_keys = {'correction'}
        self.__dict__.update((k, v) for k, v in kwargs.items() if k in allowed_keys)

    def inference_run(self):
        """
        Creates the worker processes and stores them in a pool to later collect the results.
        It divides the dataset among devices, rounding up, takes the maximum between the resulting size
        and the batch size to establish a lower portion limit and spawns (as opposed to fork()) one worker per 
        portion per GPU.
        """
        with ProcessPoolExecutor(max_workers=len(self.devs), mp_context=mp.get_context("spawn")) as ex:
            workerqueue = []
            jump = max(math.ceil(len(self.ses_dataset)/len(self.devs)) , self.batchsize) #Portion per GPU, min size batchsize
            for i in range(0,len(self.ses_dataset),jump):
                workerqueue.append(ex.submit(Req_worker(modelname=self.modelname
                                                    ,layers=self.layers
                                                    ,compfunc=self.compfunc
                                                    ,correction=self.correction
                                                    ,partialdata=self.ses_dataset[i:i+jump]
                                                    ,batchsize=self.batchsize
                                                    ,single_dev=self.devs[i%len(self.devs)]).worker_start))
            embeddings = [i.result() for i in workerqueue]
        return embeddings
            

class Session():
    """
    Represents a conglomerate of requests from a single user, made of different models to different devices.
    """
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.reqlist = args
    def session_run(self):
        inf_queue = [InferenceRequest(*i) for i in self.reqlist]
        output_embedding = [i.inference_run() for i in inf_queue]
        return output_embedding
            


if __name__ == "__main__":
    """
    For testing
    """
    xdataset=[]
    ses = Session(12, [123, 'bert-base-uncased', [11,12], 'cls', [{"tag":"test","text":"testing this."},
                                                            {"tag":"test2", "text":"And also this."},
                                                            {"tag":"test", "text":"this for the road"}], 16, ['cuda:1', 'cuda:2', 'cuda:0']])
    print(ses.session_run())
