import dash
from dash.dependencies import ALL, MATCH, Input, Output, State
from dash import html
import dash_daq as daq
from dash import dash_table


from difflib import get_close_matches
import requests
import json
import io
import base64
import gc

import pandas
import numpy

from nltk import tokenize

from app import app
import callbacks_inf

from protobuf_to_dict import protobuf_to_dict
from src.grpc_files import compservice_pb2, compservice_pb2_grpc
import grpc_if_methods

from callbacks_inf import request_record, tab_record

############################## CLIENT STARTUP SCRIPT ######################################################

def _modelrepo_update():
    """
    Gets a list of the models from the Hugging Face Model Hub, that contains a repository of all available models.
    Returns a list with the models ID (unique id)
    """
    print('Fetching data from HuggingFace.co...')
    models = [{'label':'Network Error'}]
    try:
        models = requests.get(f"https://huggingface.co/api/models").json()#, params = {"sort":"modelId"}).json()
        models = set(i["modelId"] for i in models)
    except ConnectionError as e:
        print(e.strerror)
    return models
def grpc_if_start(address):
    """
    Takes an address as String and returns its instance of the grpc interface class 
    """
    return grpc_if_methods.Server_grpc_if(address)

_model_list = _modelrepo_update()   #Runs at startup, gets the up to date list of models from HuggingFace

request_server = grpc_if_methods.Server_grpc_if("localhost:42001") #Instanciates the default grpc interface class




#############################################################################################################
###################################### INDEX DASH CALLBACKS #################################################
#############################################################################################################
######################
#### CLOSE TAB CC ####
######################
@app.callback (
    Output('tab-number-selector', 'options'),
    Input('main-tab', 'children')
)
def close_tab_selector(children):
    """
    Iterates through the active tabs and returns its value to the selector
    """
    return [{'label': f'Tab {key}', 'value':key} for key in tab_record.keys()]

@app.callback (
    Output('close-tab-button', 'value'),
    Input('close-tab-button','n_clicks'),
    State('tab-number-selector', 'value')
)
def close_tab(clicks,selection):
    """
    Takes the tab from the record so it will not be displayed.
    """
    if not clicks:
        raise dash.exceptions.PreventUpdate
    _tab = tab_record.pop(selection, None)
    del _tab
    gc.collect()
    return []
######################
#### MEMORY GAUGE ####
######################
@app.callback (
    Output('memory-gauge-div', 'children'),
    Input('add-request', 'n_clicks'),
    Input('request-list-table', 'children'),
    State('model-data-store', 'data'),
    State('batchsize-input', 'value')
)
def draw_memory_gauge(clicks, children, data, value):
    """
    Controls the memory gauge component. It asks the server for computing device information
    displays an estimation of total memory, and memory availabe after adding the request, 
    accounting for model memory and batchsize, for every device in the returned list.
    """
    _dev_mem = {}
    for key in request_record.keys():
        
        for device in request_record[key].devices:
            _dev_mem[device] = _dev_mem.get(device,0) + data.get(request_record[key].model,{}).get('size',0) + 0.01*value

    _lsgauge = []
    for dev in request_server.getDevices():
        _lsgauge.append(daq.Gauge(
                                    color={"gradient":True,"ranges":{"green":[0,1.5*dev.memory_total//3]
                                                                    ,"yellow":[1.5*dev.memory_total//3,2.8*(dev.memory_total//3)]
                                                                    ,"red":[2.8*(dev.memory_total//3),dev.memory_total]}},
                                    label= dev.id,
                                    min=0,
                                    max=dev.memory_total,
                                    showCurrentValue =True,
                                    units="GB",
                                    value = dev.memory_total - dev.memory_free + _dev_mem.get(dev.id,0),
                                    style= {'display':'inline-block'})
                                    )

    return _lsgauge
######################
#### REQUEST TABLE####
######################
@app.callback(
    Output('request-list-table', 'children'),
    Input('add-request', 'value'),
    Input({'type':'request-delete-button','index':ALL}, 'value')
)
def fill_request_table(value1, value2):
    """
    Controls the request table displayed, iterating through request_record, adding its data to
    the table.
    """
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
    """
    Controls the button that deletes the inference request from request_record.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    if id['index'] in request_record.keys(): #Guard for removing after running inference
        del request_record[id['index']]
    return 0

#######################
#### RUN INFERENCE ####
#######################
@app.callback(
    Output('update-layout', 'data'),
    Input('run-inference-button', 'n_clicks'),
    #State('main-tab', 'children')
)
def add_inf_tab(n_clicks):
    """
    Takes the request_record list, makes a copy and clears the original. Initialises the 
    tab and sends the inference request to the server, the proceeds to process the server
    output. Returns the list of keys mapping to tabs that hold the inference results.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    _record = request_record.copy()                         # Make a copy of the record to work with
    request_record.clear()                                  # Clear the record
    rec_keys = list(_record.keys())

    print('Sending session request...')
    response = [protobuf_to_dict(message) 
                for message in request_server.inf_session(_record ,tab_record)]
    print('Preprocessing response...')

    # REDO with enumerate
    for i in range(len(rec_keys)):
        response_embeddings = response[i]['embedding']
        _key = rec_keys[i]
        request_tab = tab_record[_key]
        request_tab.set_embeddings(response_embeddings)

        #children.append(request_tab.generate_tab())
    print('Sending to Tab.')
    return rec_keys

@app.callback(
    Output('main-tab', 'children'),
    Input('update-layout','data'),
    Input('close-tab-button', 'value'),
    Input('saved-ses-sink', 'n_clicks'),
    State('main-tab', 'children')
)
def update_layout(data, clicks, saved_clicks, children):
    """
    Updates the layout of the main page.
    """
    _keys = list(tab_record.keys())
    print(_keys)
    _children = []
    
    _children.append(children[0])
    for key in _keys:
        _children.append(tab_record[key].generate_tab())
    
    return _children

@app.callback(
    Output('inference-is-loading', 'children'),
    Input('inference-loading', 'value' )
)
def inference_loading_state(value):
    return value

@app.callback(
    Output('run-inference-button', 'disabled'),
    Input('run-inference-button', 'n_clicks')
)
def disable_run_inference_button(clicks):
    if not clicks:
        raise dash.exceptions.PreventUpdate
    return 'disabled'
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
    return_models = get_close_matches(search_value, _model_list, 10,0.3)
    options = [{'label': i, 'value': i} for i in return_models]
    return options

### DOWNLOAD Transformer ###
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
    """
    Control for the download button
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return request_server.downloadModel(value).completed
### DOWNLOAD Static ###
@app.callback(
    Output('download-static-model-button', 'disabled'),
    Input('static-model-dropdown', 'value')
)
def activate_downloadStatic_button(value):
    """
    Activates the Download button upon selection of a Model from the repository
    """
    disabled = False
    if not value:
        disabled=True
    return disabled
@app.callback(
    Output('download-static-model-button', 'value'),
    Input('download-static-model-button', 'n_clicks'),
    State('static-model-dropdown', 'value')
)
def download_model(n_clicks, value):
    """
    Control for the static download button
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return request_server.downloadStatic(value).completed 
### STATIC DROPDOWN ###
@app.callback(
    Output('static-model-data-store', 'data'),
    Input('download-static-model-button', 'value'),
    Input('delete-static-model-button', 'value'),
    Input('static-block-models', 'label')
)
def update_static_model_list_store(valuedown, valuedel, label):
    """
    Probes the server for the list of downloaded models.
    """
    return request_server.getStaticModels()

@app.callback(
    Output('static-block-models', 'options'),
    Input('static-model-data-store','data'),
    State('static-model-data-store', 'data')
)
def update_static_model_list(valuedown, data):
    """
    Controls the selector that displays the downloaded models.
    """
    if (not valuedown == False):
        options = [{'label': i, 'value': i} for i in [str(i) for i in data]] 
    return options
@app.callback(
    Output('static-block-models','value'),
    Input('delete-static-model-button', 'n_clicks')
)
def clear_static_model_dropdown_selection(n_clicks):
    return None

### DROPDOWN ###
@app.callback(
    Output('model-data-store', 'data'),
    Input('download-model-button', 'value'),
    Input('delete-model-button', 'value'),
    Input('block-models', 'label')
)
def update_model_list_store(valuedown, valuedel, label):
    """
    Probes the server for the list of downloaded models.
    """
    return request_server.getModels()

@app.callback(
    Output('block-models', 'options'),
    Input('model-data-store','data'),
    State('model-data-store', 'data')
)
def update_model_list(valuedown, data):
    """
    Controls the selector that displays the downloaded models.
    """
    if (not valuedown == False):
        options = [{'label': i, 'value': i} for i in [str(i) for i in data]] 
    return options
@app.callback(
    Output('block-models','value'),
    Input('delete-model-button', 'n_clicks')
)
def clear_model_dropdown_selection(n_clicks):
    return None

@app.callback(
    Output('hf-link', 'href'),
    Input('block-models', 'value')
)
def link_to_model_page_href(value):
 
    if not value:
        raise dash.exceptions.PreventUpdate
    return f'http://huggingface.co/{value}'

@app.callback(
    Output('hf-link', 'children'),
    Input('block-models', 'value')
)
def link_to_model_page(value):
    """
    Generates and displays a link to the model page in HuggingFace repository.
    """
    if not value:
        return html.P()
    return html.P(f'http://huggingface.co/{value}')

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
    """
    Controls the delete model button.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return request_server.deleteModel(value).completed
### DELETE STATIC ###
@app.callback(
    Output('delete-static-model-button', 'style'),
    Input('static-block-models', 'value'),
    State('delete-static-model-button', 'style')
)
def activate_static_delete_button(value, style):
    """
    Activates the delete button upon selection of a Model
    """
    if not value and not 'visibility' in style:
        style['visibility']='hidden'
    if value and 'visibility' in style:
        style.pop('visibility')
    return style
@app.callback(
    Output('delete-static-model-button', 'value'),
    Input('delete-static-model-button', 'n_clicks'),
    State('static-block-models', 'value')
)
def delete_model(n_clicks, value):
    """
    Controls the delete model button.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    return request_server.deleteStatic(value).completed
### LAYER SLIDER ###
@app.callback(
    Output('layer-slider', 'disabled'),
    Input('block-models', 'value')
)
def activate_slider(value):
    """
    Makes the slider interactive.
    """
    return True if not value else False

@app.callback(
    Output('layer-slider', 'max'),
    Input('block-models', 'value'),
    State('model-data-store', 'data')
)
def update_layer_slider(value, data):
    """
    Updates the slider layout with the number of layers given the model selected.
    """
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
    Processes the data coming from the upload frame,
    and shows is.
    If txt, it uses the NLTK librarty to split the sentences.
    No separator is specified for csv, json and xls; the dialect engine is 
    trusted for that.
    """
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    df = pandas.DataFrame({'A' : []})                                       # Fallback value

    try:
        if 'txt' in filename:
            df = pandas.DataFrame(tokenize.sent_tokenize(decoded.decode('utf-8-sig')))
        if 'csv' in filename:
            df = pandas.read_csv(io.StringIO(decoded.decode('utf-8', errors='replace'))) #sep
        if 'json' in filename:
            df = pandas.read_json(io.StringIO(decoded.decode('utf-8')), lines=True) 
        if 'xls' in filename:
            df = pandas.read_excel(io.BytesIO(decoded))

    except Exception as e:
        print(f'Issue converting file types : {e}')   
    return df
    
def make_table(contents, df, filename):
    """
    Generates the code for the datatable showing a sample size.
    """
    if  df.empty:
        return html.Div([ 'There was an error processing the file.'])

    if  not df.empty:
        table_fragment =html.Div([
        html.H4('Select the column with the sentences to process:'),
        html.H5(filename),
        dash_table.DataTable(
            id = 'dataset-table',
            data = df.head(10).to_dict('records'),
            columns = [{'name':i, 'id':i, 'selectable':True} for i in df.columns],
            style_table={'overflowY': 'scroll'},
            style_cell={'textAlign':'left'},
            column_selectable='multi'
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
    """
    Outputs the table generated, given the dataset.

    Args:
        list_of_contents(b64): The file in b64 format
        list_of_names(string): The name of the file
    
    """
   
    if not list_of_names:
        return []
    
    if not list_of_contents:
        return [html.P('There was an error processing the file. Make sure the size is under 1GB.')]
        
    children = []  
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
@app.callback(
    Output('file-ul-req', 'filename'),
    Input('add-request', 'n_clicks')
)
def clear_names(value):
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
    """
    Probes the server for the available computing devices.
    """
    options = [{'label': f"{dev.device_name} ID: {dev.id}", 
                'value': dev.id} for dev in request_server.getDevices()]
    return options


#####################
#### ADD REQUEST ####
#####################
@app.callback(
    Output('add-request', 'value'),
    Input('add-request', 'n_clicks'),
    State('block-models', 'value'),
    State('static-block-models', 'value'),
    State('layer-slider', 'value'),
    State('file-ul-req', 'filename'),
    State('file-ul-req', 'contents'),
    State('dataset-table', 'selected_columns'),
    State('batchsize-input', 'value'),
    State('block-composition', 'value'),
    State('select-devices', 'value'),
    State('tab-count', 'data'),
    State('model-type-tab', 'value')

)
def add_request(n_clicks
                ,model
                ,static_model
                ,layers
                ,filename
                ,contents
                ,sel_columns
                ,valuebatch
                ,valuecomp
                ,valuedev
                ,tab_index
                ,model_repr):
    """
    Collects all the information given by the user from the different page elements
    and initialises the request object, storing it in request_record.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not ((model or static_model) and sel_columns and valuecomp and valuedev):
        raise dash.exceptions.PreventUpdate

    _df = parse_file_ul(contents,filename)
    
    _request = callbacks_inf.inference_tab(index=tab_index
                                            ,model=model if 'transformer' in model_repr else static_model
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
    Input('select-devices', 'value'),
    Input('static-block-models', 'value')
)
def activate_add_request(model
                        ,sel_columns
                        ,valuecomp
                        ,valuedev
                        ,static_model):
    _state= False
    if not ((model or static_model) and sel_columns and valuecomp and valuedev):
        return True
    return _state

@app.callback(
    Output('tab-count', 'data'),
    Input('add-request', 'value'),
    Input('saved-ses-sink', 'n_clicks'),
    State('tab-count', 'data')
)
def update_tab_index(value, children, data):
    if not value and not children:
        raise dash.exceptions.PreventUpdate
    return data+1
    
############################
#### LOAD SAVED SESSION ####
############################
@app.callback(
    Output('saved-ses-sink', 'n_clicks' ),
    Input('load-saved-session', 'filename'),
    State('load-saved-session', 'contents'),
    State('tab-count', 'data')
)
def load_saved(filename, contents, index):
    if not filename:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    
    _saved_session = None
    try:
        if 'aspk' in filename:
            _saved_session = json.loads(decoded.decode('utf-8-sig'))
            for key in _saved_session.keys():
                numpy.array(_saved_session[key]['value'], dtype='float32')


    except Exception as e:
        print(f'Issue converting file types: {e}')  

    # RETOCAR: Añadir objeto en clase contenedora o pasar diccionario por parámetro
    print(_saved_session.keys())
    key_base = list(_saved_session.keys())[0]
    print(index)
    saved_session = callbacks_inf.inference_tab(index=index
                                            ,model=_saved_session[key_base]['model']
                                            ,dataset=pandas.DataFrame(_saved_session[key_base]['sentences']
                                                                    ,columns={_saved_session[key_base]['column']})
                                            ,filename=_saved_session[key_base]['filename']
                                            ,text_column=[_saved_session[key_base]['column']]
                                            ,layer_low=_saved_session[key_base]['layers'][0]
                                            ,layer_up=_saved_session[key_base]['layers'][1]
                                            ,comp_func=_saved_session[key_base]['composition']
                                            ,embeddings=_saved_session[key_base]['value']
                                            ,batchsize=0
                                            ,devices=['cpu'])
    
    for key in list(_saved_session.keys())[1:]:
        inf_data = callbacks_inf.Inference(index=index
                                            ,model=_saved_session[key]['model']
                                            ,dataset=pandas.DataFrame(_saved_session[key]['sentences']
                                                                    ,columns={_saved_session[key]['column']})
                                            ,filename=_saved_session[key]['filename']
                                            ,text_column=_saved_session[key]['column']
                                            ,layer_low=_saved_session[key]['layers'][0]
                                            ,layer_up=_saved_session[key]['layers'][1]
                                            ,comp_func=_saved_session[key]['composition']
                                            ,embeddings=_saved_session[key]['value']
                                            ,batchsize=0
                                            ,devices=['cpu'])
        saved_session.append(inf_data)

    tab_record[index] = saved_session
    return 1

@app.callback(
    Output('load-saved-session', 'filename'),
    Input('saved-ses-sink', 'n_clicks')
)
def reset_loading_saved(n_clicks):
    return None

@app.callback(
    Output('load-saved-session', 'contents'),
    Input('saved-ses-sink', 'n_clicks')
)
def reset_loading_saved(n_clicks):
    return None