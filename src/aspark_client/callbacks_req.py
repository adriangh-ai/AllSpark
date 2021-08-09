import dash
from dash.dependencies import ALL, Input, MATCH, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import nltk
import pandas
import google.protobuf as pb
from google.protobuf.json_format import Parse
from google.protobuf.json_format import MessageToJson

from difflib import get_close_matches
import requests
import os
import json
import io
import base64

from nltk import tokenize
from requests.sessions import session
from torch._C import device

from app import app
import callbacks_inf

import grpc
from src.grpc_files import compservice_pb2, compservice_pb2_grpc
import grpc_if_methods


############################## CLIENT STARTUP SCRIPT ######################################################

def _modelrepo_update():
    """
    Gets a list of the models from the Hugging Face Model Hub, that contains a repository of all available models.
    Returns a list with the models ID (unique id)
    """
    models = [{'label':'Network Error'}]
    try:
        models = requests.get(f"https://huggingface.co/api/models", params = {"sort":"modelId"}).json()
        models = set(i["modelId"] for i in models)
    except ConnectionError as e:
        print(e.strerror)
    return models

_model_list = _modelrepo_update()   #Runs at startup, gets the up to date list of models from HuggingFace

request_server = grpc_if_methods.Server_grpc_if("localhost:42001") #Instanciates the grpc interface class

request_record = {}    # Record where the requests are temporary stored
tab_record = {}        # Record where the Inference Tabs are stored


#############################################################################################################
###################################### INDEX DASH CALLBACKS #################################################
#############################################################################################################

######################
#### REQUEST TABLE####
######################
@app.callback(
    Output('request-list-table', 'children'),
    Input('add-request', 'value'),
    Input({'type':'request-delete-button','index':ALL}, 'value')
)
def fill_request_table(value1, value2):
    _children = []
    if len(request_record) > 0:
        _children.append(
            html.Tr(children=[
                html.Th('Model'),
                html.Th('Layers'),
                html.Th('Dataset'),
                html.Th('Column'),
                html.Th('Comp. F.'),
                html.Th('Batchsize'),
                html.Th('Devices'),
                html.Th('')
            ])
        )
        for request in request_record.values():
            _children.append(
                html.Tr(children=[
                    html.Td(str(request.model)),
                    html.Td(f'{request.layer_low} - {request.layer_up}'),
                    html.Td(str(request.filename)),
                    html.Td(str(request.text_column)),
                    html.Td(str(request.comp_func)),
                    html.Td(str(request.batchsize)),
                    html.Td(str(request.devices)),
                    html.Td(children=[
                        html.Button('REMOVE'
                                    ,id= {'type':'request-delete-button','index': request.index})
                    ])
                ])
            )
        _children.append(
            html.Div(
                html.Button('RUN INFERENCE', id='run-inference-button')
            )
        )
    return _children
@app.callback(
    Output({'type':'request-delete-button','index':MATCH}, 'value'),
    Input({'type':'request-delete-button','index':MATCH}, 'n_clicks'),
    State({'type':'request-delete-button','index':MATCH}, 'id')
)
def remove_from_request_list(n_clicks, id):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    del request_record[id['index']]
    return 0

#######################
#### RUN INFERENCE ####
#######################
@app.callback(
    Output('main-tab', 'children'),
    Input('run-inference-button', 'n_clicks'),
    State('main-tab', 'children')
)
def add_inf_tab(n_clicks, children):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    _record = request_record.copy()
    request_record.clear()

    session_pb = compservice_pb2.Session()
    _request_list = []

    for i in list(_record.keys()):
        tab_record[i] = _record.pop(i)

        request = compservice_pb2.Request()
        request.model = tab_record[i].model
        request.layer_low= tab_record[i].layer_low
        request.layer_up = tab_record[i].layer_up
        request.comp_func = tab_record[i].comp_func
        request.batchsize = tab_record[i].batchsize
        request.devices.name.extend(tab_record[i].devices)

        _dataset = tab_record[i].dataset[tab_record[i].text_column]
        #_dataset['posicion'] = _dataset.reset_index().index
        #_dataset = _dataset.to_json(orient='records')
        _message = []
        
        for index,row in _dataset.iterrows():
            _message.append(request.data.Sentence(position=index
                                                ,sentence=row[tab_record[i].text_column].item()))

        request.data.sentence_data.extend(_message)

        _request_list.append(request)
    
    session_pb.request.extend(_request_list)

    print(request_server.inf_session(session_pb))

        
        
        #_message = Parse(i, request_pb.Sentence())
        
    print(_message[0])

    new_tab = dcc.Tab(label = 'Inference',className='main-tabs', children=[
                        html.Div( id={'type':'inf-tab', 'index': n_clicks },children=[
                            html.P('this is al there is inside'),
                            html.Button('will this work', id= {'type':'button-new'
                                                                ,'index': n_clicks }, n_clicks=0, name='hmm')
                        ])
            ])
    children.append(new_tab)
    return children

#########################
#### MODEL SELECTION ####
#########################
@app.callback(
    Output('model-dropdown', 'options'),
    Input('model-dropdown', 'search_value'),
    State('model-dropdown', 'options'))
def model_search(search_value, options):
    """
    Gets updated on user keyboard input and searches the closest matches amont the
    entries in the Hugging Face model list.
    Returns the closest matches parsed for the Dash Dropdown Menu
    """
    if not search_value:
        raise dash.exceptions.PreventUpdate
    return_models = get_close_matches(search_value, _model_list, 10,0.5)
    options = [{'label': i, 'value': i} for i in return_models]
    return options

### DOWNLOAD ###
@app.callback(
    Output('download-model-button', 'disabled'),
    Input('model-dropdown', 'value')
)
def activate_download_button(value):
    """
    Activates the Download button upon selection of a Model from the repository
    """
    disabled = False
    if not value:
        disabled=True
    return disabled
@app.callback(
    Output('download-model-button', 'value'),
    Input('download-model-button', 'n_clicks'),
    State('model-dropdown', 'value')
)
def download_model(n_clicks, value):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return request_server.downloadModel(value).completed
    
### DROPDOWN ###
@app.callback(
    Output('model-data-store', 'data'),
    Input('download-model-button', 'value'),
    Input('delete-model-button', 'value'),
    Input('block-models', 'label')
)
def update_model_list_store(valuedown, valuedel, label):
    return request_server.getModels()

@app.callback(
    Output('block-models', 'options'),
    Input('model-data-store','data'),
    State('model-data-store', 'data')
)
def update_model_list(valuedown, data):
    if (not valuedown == False):
        options = [{'label': i, 'value': i} for i in [str(i) for i in data]] 
    return options
@app.callback(
    Output('block-models','value'),
    Input('delete-model-button', 'n_clicks')
)
def clear_model_dropdown_selection(n_clicks):
    return None

### DELETE ###
@app.callback(
    Output('delete-model-button', 'style'),
    Input('block-models', 'value'),
    State('delete-model-button', 'style')
)
def activate_delete_button(value, style):
    """
    Activates the delete button upon selection of a Model
    """
    if not value and not 'visibility' in style:
        style['visibility']='hidden'
    if value and 'visibility' in style:
        style.pop('visibility')
    return style
@app.callback(
    Output('delete-model-button', 'value'),
    Input('delete-model-button', 'n_clicks'),
    State('block-models', 'value')
)
def delete_model(n_clicks, value):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return request_server.deleteModel(value).completed

### LAYER SLIDER ###
@app.callback(
    Output('layer-slider', 'disabled'),
    Input('block-models', 'value')
)
def activate_slider(value):
    return True if not value else False

@app.callback(
    Output('layer-slider', 'max'),
    Input('block-models', 'value'),
    State('model-data-store', 'data')
)
def update_layer_slider(value, data):
    if not value:
        raise dash.exceptions.PreventUpdate
    return data[value]['layers'] 

@app.callback(
    Output('layer-slider', 'marks'),
    Input('layer-slider', 'max')
)
def update_layer_slider_marks(max):
    if not max:
        raise dash.exceptions.PreventUpdate
    return {i:{'label':str(i)} for i in range(1,max+1)}
@app.callback(
    Output('layer-slider', 'value'),
    Input('layer-slider','max')
)
def update_layer_slider_preselection(max):
    return [max,max]

########################################
#####  DATASET SELECTION CALLBACKS #####
########################################
def parse_file_ul(contents, filename):
    """
    Processes a sample size of the data coming from the upload frame,
    and shows is.
    If txt, it uses the NLTK librarty to split the sentences.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    df = pandas.DataFrame({'A' : []})                                       # Fallback value
    table_fragment = html.Div([ 'There was an error processing the file.']) # Fallback return

    try:
        if 'txt' in filename:
            df = pandas.DataFrame(tokenize.sent_tokenize(decoded.decode('utf-8')))
        if 'csv' in filename:
            df = pandas.read_csv(io.StringIO(decoded.decode('utf-8', errors='replace')), nrows=10, sep='\s+')
        if 'json' in filename:
            df = pandas.read_json(io.StringIO(decoded.decode('utf-8'))) #lines? ^ separator?
        if 'xls' in filename:
            df = pandas.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(e)   
    return df
    
def make_table(contents, df, filename):
    if  df.empty:
        return html.Div([ 'There was an error processing the file.'])
    if  not df.empty:
        table_fragment =html.Div([
        html.P('Select the column with the sentences to process'),
        html.H5(filename),
        dash_table.DataTable(
            id = 'dataset-table',
            data = df.head(10).to_dict('records'),
            columns = [{'name':i, 'id':i, 'selectable':True} for i in df.columns],
            style_table={'overflowY': 'scroll'},
            column_selectable='single'
        ),
        html.Div('Raw Content'),
        html.Pre(contents[0:200] + '...', style={
            'whiteSpace': 'pre-wrap',
            'wordBreak': 'break-all'
        })
    ])
    return table_fragment

@app.callback( 
    Output('display-ul-data','children'),
    Input('file-ul-req', 'contents'),
    State('file-ul-req', 'filename'))
def update_data_ul(list_of_contents, list_of_names):
    children=[]
    if (list_of_contents is not None) and not (len(list_of_contents)==0):
        print(list_of_names)
        print(type(list_of_contents))
        print(len(list_of_contents))
        children = [
            #parse_file_ul(c,d) for c,d in zip(list_of_contents, list_of_names)]
            make_table(list_of_contents,
                        parse_file_ul(list_of_contents, list_of_names),
                        list_of_names)]
    return children
@app.callback(
    Output('file-ul-req', 'contents'),
    Input('add-request', 'value')
)
def clear_content(value):
    return None

###############################
#### COMPOSITION SELECTION ####
###############################
""" @app.callback(
    Output('block-composition', 'options'),
    Input('block-model', 'value'),
    State('block-composition', 'options')
)
def adjust_comp_selection_to_model(value,options):
    
    pass #TODO """


###############################
#### COMPUTING DEVICES SEL ####
###############################
@app.callback(
    Output('select-devices', 'options'),
    Input('select-devices', 'options')
)
def update_devices_list(options):
    options = [{'label': f"{dev.device_name} ID: {dev.id}", 
                'value': dev.id} for dev in [i for i in request_server.getDevices()]]
    return options

""" @app.callback(
    Output('select-devices', 'value'),
    Input('select-devices', 'value'),
    State('select-devices', 'value')
)
def update_devices_list(valuea, valueb):
    if not valuea:
        raise dash.exceptions.PreventUpdate
    print
    if "cpu" in valuea:
        value = ['cpu']
    return value """

#####################
#### ADD REQUEST ####
#####################
@app.callback(
    Output('add-request', 'value'),
    Input('add-request', 'n_clicks'),
    State('block-models', 'value'),
    State('layer-slider', 'value'),
    State('file-ul-req', 'filename'),
    State('file-ul-req', 'contents'),
    State('dataset-table', 'selected_columns'),
    State('batchsize-input', 'value'),
    State('block-composition', 'value'),
    State('select-devices', 'value'),
    State('tab-count', 'data')
)
def add_request(n_clicks
                ,model
                ,layers
                ,filename
                ,contents
                ,sel_columns
                ,valuebatch
                ,valuecomp
                ,valuedev
                ,tab_index):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not (model and sel_columns and valuecomp and valuedev):
        raise dash.exceptions.PreventUpdate

    _df =parse_file_ul(contents,filename)
    _request = callbacks_inf.inference_tab(index=tab_index
                                            ,model=model
                                            ,dataset=_df
                                            ,filename=filename
                                            ,text_column=sel_columns
                                            ,layer_low=layers[0]
                                            ,layer_up=layers[1]
                                            ,comp_func=valuecomp
                                            ,batchsize=valuebatch
                                            ,devices=valuedev)
    request_record[tab_index]= _request
    return tab_index + 1

@app.callback(
    Output('add-request', 'n_clicks'),
    Input('add-request', 'value')
) 
def disable_add_request(value):
    return None  

@app.callback(
    Output('add-request', 'disabled'),
    Input('block-models', 'value'),
    Input('dataset-table', 'selected_columns'),
    Input('block-composition', 'value'),
    Input('select-devices', 'value')
)
def activate_add_request(model
                        ,sel_columns
                        ,valuecomp
                        ,valuedev):
    _state= False
    if not (model and sel_columns and valuecomp and valuedev):
        return True
    return _state

@app.callback(
    Output('tab-count', 'data'),
    Input('add-request', 'value')
)
def update_tab_index(value):
    if not value:
        raise dash.exceptions.PreventUpdate
    return value
    
#### GAUGE

##############################pariah

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