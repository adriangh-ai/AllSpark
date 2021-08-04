import dash
from dash.dependencies import Input, MATCH, Output, State
import dash_core_components as dcc
import dash_html_components as html

from difflib import get_close_matches
import requests
import os
import json

from app import app
import grpc
from src.grpc_files import compservice_pb2, compservice_pb2_grpc



channel = grpc.insecure_channel('localhost:42001')
stub = compservice_pb2_grpc.compserviceStub(channel)

def _modelrepo_update():
    models = [{'label':'Network Error'}]
    try:
        models = requests.get(f"https://huggingface.co/api/models", params = {"sort":"modelId"}).json()
        models = set(i["modelId"] for i in models)
    except ConnectionError as e:
        print(e.strerror)
    return models

_model_list = _modelrepo_update()


@app.callback(
    Output('model-dropdown', 'options'),
    Input('model-dropdown', 'search_value'),
    State('model-dropdown', 'options'))
def model_search(search_value, options):
    if not search_value:
        raise dash.exceptions.PreventUpdate
    return_models = get_close_matches(search_value, _model_list, 10,0.5)
    options = [{'label': i, 'value': i} for i in return_models]
    return options
    

@app.callback(
    Output('main-tab', 'children'),
    Input('add-tab', 'n_clicks'),
    State('main-tab', 'children'))
def add_inf_tab(n_clicks, children):

    if not n_clicks == 0:
        new_tab = dcc.Tab(label = 'Inference',className='main-tabs', children=[
                            html.Div( id={'type':'inf-tab', 'index': n_clicks },children=[
                                html.P('this is al there is inside'),
                                html.Button('will this work', id= {'type':'button-new', 'index': n_clicks }, n_clicks=0, name='hmm')
                            ])
                ])
        children.append(new_tab)
    return children

@app.callback(
    Output({'type': 'inf-tab', 'index': MATCH}, 'children'),
    Input({'type': 'button-new', 'index': MATCH}, "n_clicks"),
    State({'type': 'inf-tab', 'index': MATCH}, 'children')
)
def add_content(n_clicks, children):
    newp = html.P("testing if this works OMG")
    children = [newp]
    return children

if __name__ == '__main__':
   print(model_search("bert", _model_list))