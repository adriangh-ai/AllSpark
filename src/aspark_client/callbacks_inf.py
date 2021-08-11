from dash.dependencies import Input, MATCH, Output, State
import dash
import dash_core_components as dcc
import dash_html_components as html

from collections import OrderedDict
import numpy as np

from app import app

request_record = {}         # Record where the requests are temporary stored
tab_record = OrderedDict()  # Record where the Inference Tabs are stored

class inference_tab():
    """
    This class contains the details of the Inference Tab it represents,
    identified by the index argument.
    """
    def __init__(self
                ,index
                ,model
                ,dataset
                ,filename
                ,text_column
                ,layer_low
                ,layer_up
                ,comp_func
                ,batchsize
                ,devices):

        self.index=index
        self.model=model
        self.dataset = dataset
        self.filename= filename
        self.text_column=text_column
        self.layer_low = layer_low
        self.layer_up = layer_up
        self.comp_func = comp_func
        self.batchsize = batchsize
        self.devices = devices
        self.embeddings = []
    
    def set_embeddings(self, embed_dict):
        embed_stack = []
        for sentence in embed_dict:
            embed_stack.append(np.array(sentence['value'], dtype=np.float32))
        self.embeddings = np.concatenate(embed_stack)
    
    def generate_tab(self):
        new_tab = dcc.Tab(label = 'Inference',className='main-tabs', children=[
                        html.Div( id={'type':'inf-tab', 'index': self.index },children=[
                            html.P('this is al there is inside'),
                            html.Button('will this work', id= {'type':'button-new'
                                                                ,'index': self.index }, n_clicks=0, name='hmm')
                        ])
            ])
        return new_tab