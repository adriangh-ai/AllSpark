import dash
from dash.dependencies import Input, MATCH, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import pandas

from difflib import get_close_matches
import requests
import os
import json
import io
import base64

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
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
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
    State({'type': 'inf-tab', 'index': MATCH}, 'children'),
    State({'type': 'inf-tab', 'index': MATCH}, 'id')
)
def add_content(n_clicks, children,id):
    if n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    newp = html.P(f"testing if this works OMG index is{id['index']}")
    children = [newp]
    return children

if __name__ == '__main__':
   print(model_search("bert", _model_list))


#####  DATASET SELECTION CALLBACKS #####
def parse_file_ul(contents, filename):
    print(filename)
    
    content_type, content_string = contents.split(',')
    print(type(content_type))
    
    decoded = base64.b64decode(content_string)

    try:
        if 'csv' in filename:
            df = pandas.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        if 'json' in filename:
            df = pandas.read_json(
                io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        print(e)
        return html.Div([ 'There was an error processing the file.'])
    
    return html.Div([
        html.H5(filename),
        dash_table.DataTable(
            data = df.head(10).to_dict('records'),
            columns = [{'name':i, 'id':i} for i in df.columns],
            style_table={'overflowY': 'scroll'}
        ),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])

@app.callback( 
    Output('display-ul-data','children'),
    Input('file-ul-req', 'contents'),
    State('file-ul-req', 'filename'))
def update_data_ul(list_of_contents, list_of_names):
    if list_of_contents is not None:
        print(list_of_names)
        children = [
            #parse_file_ul(c,d) for c,d in zip(list_of_contents, list_of_names)]
            parse_file_ul(list_of_contents, list_of_names)]
        return children