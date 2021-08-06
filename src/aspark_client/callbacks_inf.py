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
    def __init__(self, index):
        self.index=index
        self.model=[]
        self.dataset = []
        self.layer_low = 12
        self.layer_up = 12
        self.comp_fund = 'cls'
        self.batchsize = 16
        self.devices = ['cpu']