import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from nltk import data

class Req_worker():
    def __init__(self, modelname, layers, compfunc, correction, parcialdata, batchsize:int, single_dev):
        self.modelname = modelname
    def worker_start(self):
        return "kinda worked"

class InferenceRequest():
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
        with ProcessPoolExecutor(max_workers=len(self.devs), mp_context=mp.get_context("spawn")) as ex:
            
            workerqueue = [ex.submit(Req_worker(modelname=self.modelname
                                                ,layers=self.layers
                                                ,compfunc=self.compfunc
                                                ,correction=self.correction
                                                ,parcialdata=self.ses_dataset
                                                ,batchsize=self.batchsize
                                                ,single_dev=i).worker_start) for i in self.devs]
            embeddings = [i.result for i in workerqueue]
        return embeddings
            

class Session():
    def __init__(self, id, *args, **kwargs):
        self.id = id
        self.reqlist = args
    def session_run(self):
        print(self.reqlist)
        inf_queue = [InferenceRequest(*i) for i in self.reqlist]
        output_embedding = [i.inference_run() for i in inf_queue]
        return output_embedding
            


if __name__ == "__main__":
    xdataset=[]
    ses = Session(12, [123, 'bert-base-uncased', 'cls', xdataset, ['cuda:1', 'cuda:2']])
    print(ses.session_run())
