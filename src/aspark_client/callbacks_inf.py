from dash.dependencies import Input, MATCH, Output, State
import dash
import dash_core_components as dcc
import dash_html_components as html

from app import app

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