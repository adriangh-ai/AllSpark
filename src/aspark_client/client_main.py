import sys
import socket
from pathlib import Path

sys.path.append(str(Path(__file__).parents[2])) #Work Path

import dash
from dash import dcc
from dash import html
#import dash_uploader as du

from app import app

#du.configure_upload(app, f'{str(Path(__file__).parents[2])}/tmp')

import callbacks_req
import callbacks_inf

from waitress import serve


app.title = "AllSpark"
app.layout = html.Div(id='app-index',children=[
    dcc.Store(id='tab-count', storage_type='session', data=1),
    dcc.Loading(children=[
    dcc.Store(id='update-layout'),
    html.Div(
        className="app-index-header",
        children=[
            html.Div(className="app-header--title", children=[
                html.H1("AllSpark", title="AllSpark", style={'display':'inline-block' }),
                html.Button('Close Tab', id='close-tab-button' ,style={'display':'inline-block!important'
                                                                        ,'position':'relative'
                                                                        ,'float':'right'}),
                dcc.Dropdown(id='tab-number-selector'   ,options=[]
                                                        ,multi=False
                                                        ,style={'width':'100px'
                                                                ,'display':'inline-block'
                                                                ,'position':'relative'
                                                                ,'float':'right'
                                                                 })
            ])
        ]
    )]),
    html.Div(
        children = [dcc.Tabs(id="main-tab",className="main-tabs", value='tab-1', children=[
            dcc.Tab(label='Session', value='tab-1', className='main-tabs', children=[
                html.Div(id = "block-req-list", className='tab-block', children=[
                    html.Div(id='memory-gauge-div', children = [
                    ],
                    style={'text-align': 'center'}
                    ),
                    html.Div( children=[
                    html.Table(id='request-list-table', children=[])
                    ]),
                    html.Div(id='inference-is-loading')
                ]),
                    html.Div(   children=[
                        html.Div(className='tab-block', children=[
                            html.H3('Composition Method'),
                            html.Div([
                                        "Choose Composition Method",
                                        dcc.RadioItems(id = "block-composition",
                                            options=[
                                                {'label': 'Sum', 'value': 'sum'},
                                                {'label': 'Average', 'value': 'avg'},
                                                {'label': 'CLS Token', 'value': 'cls'},
                                                {'label': 'F Joint', 'value': 'f_joint'},
                                                {'label': 'F Ind', 'value': 'f_ind'},
                                                {'label': 'F Inf', 'value': 'f_inf'}
                                            ],
                                            value='sum',
                                            labelStyle={'display':'block!important'}
                                        )
                            ])
                        ], style={'display':'inline-block', 'width':'20%'}),
                        html.Div(id = "block-model", className='tab-block', children=[
                            #REINDENT
                            dcc.Tabs(id='model-type-tab',className='main-tabs',value='transformer', children = [
                                dcc.Tab(label= 'Transformer', value='transformer',className='main-tabs', children= [
                            html.H3('Model Selection', style={"text-align":"centered"}),
                            html.Div([ html.Label('HugginFace Model Repository'),
                                    html.Div([
                                        dcc.Dropdown(id='model-dropdown', className="dropdown-menu", options=[],
                                            multi=False,
                                            style={"min-width":"50%", "display":"inline-block"}
                                        ),
                                        html.Div( children= [
                                        dcc.Loading(id = 'download-loading', children = [
                                            html.Button(id='download-model-button',
                                                children=["Download"],
                                                disabled=True,
                                                
                                            ) ]) ,  ], style={"display": "inline-block","position":"absolute"})
                                    ]),
                                    html.Div(children=[
                                        html.A(id='hf-link',href='', target='_blank',
                                        children=[
                                            html.P(title='Link to model page.')
                                        ]),
                                        html.P("Select Model"),
                                        dcc.Store(id='model-data-store'),
                                        dcc.RadioItems( id='block-models',
                                            options=[],
                                            labelStyle={'display':'block!important'}
                                        ),
                                        html.Button(id="delete-model-button",children=["Delete"],
                                            style={"display": "inline-block","position":"relative",'visibility':'hidden'},
                                        ),
                                    ]),
                                    
                                    html.Div(id="layer-div", children=[
                                        html.Label('Layer selector'),
                                        dcc.RangeSlider(id='layer-slider',
                                            min=1,
                                            max=12,
                                            step=1,
                                            value=[12,12],
                                            allowCross=False,
                                            updatemode='mouseup',
                                            disabled=True
                                        )
                                    ])
                            ])]),
                            dcc.Tab(label='Static Rep', value='static-rep', className='main-tabs', children = [
                                 html.H3('Model Selection', style={"text-align":"centered"}),
                            html.Div([ html.Label('Gensim Respository'),
                                    html.Div([
                                        dcc.Dropdown(id='static-model-dropdown', className="dropdown-menu", options=[
                                            {'label': 'fasttext-wiki-news-subwords-300','value': 'fasttext-wiki-news-subwords-300'},
                                            {'label':'conceptnet-numberbatch-17-06-300', 'value': 'conceptnet-numberbatch-17-06-300'},
                                            {'label':'word2vec-ruscorpora-300', 'value': 'word2vec-ruscorpora-300'},
                                            {'label':'word2vec-google-news-300', 'value': 'word2vec-google-news-300'},
                                            {'label':'glove-wiki-gigaword-50', 'value': 'glove-wiki-gigaword-50'},
                                            {'label':'glove-wiki-gigaword-100', 'value': 'glove-wiki-gigaword-100'},
                                            {'label':'glove-wiki-gigaword-200', 'value': 'glove-wiki-gigaword-200'},
                                            {'label':'glove-wiki-gigaword-300', 'value': 'glove-wiki-gigaword-300'},
                                            {'label':'glove-twitter-25', 'value': 'glove-twitter-25'},
                                            {'label':'glove-twitter-50', 'value': 'glove-twitter-50'},
                                            {'label':'glove-twitter-100', 'value': 'glove-twitter-100'},
                                            {'label':'glove-twitter-200', 'value': 'glove-twitter-200'}
                                        ],
                                            multi=False,
                                            style={"min-width":"50%", "display":"inline-block"}
                                        ),
                                        html.Div( children= [
                                            dcc.Loading(id = 'download-static-loading', children = [
                                                html.Button(id='download-static-model-button',
                                                    children=["Download"],
                                                    disabled=True,) 
                                            ])]
                                            ,style={"display": "inline-block","position":"absolute"}
                                        ),
                                  
                                    html.Div(children=[
                                         html.A(href='https://github.com/RaRe-Technologies/gensim-data/#Models',
                                            target='_blank',
                                            children=[
                                                html.P('Gensim Model Info')
                                        ]),
                                        html.P("Select Model"),
                                        dcc.Store(id='static-model-data-store'),
                                        dcc.RadioItems( id='static-block-models',
                                            options=[],
                                            labelStyle={'display':'block!important'}
                                        ),
                                        html.Button(id="delete-static-model-button",children=["Delete"],
                                            style={"display": "inline-block","position":"relative",'visibility':'hidden'},
                                        ),
                                    ]),
                                       
                                    
                            ]) 
                            ]) 
                            ])
                            ])   #REINDENT
                            
                        ], style={'display':'inline-block', 'width':'60%'}),
                        html.Div(id = "block-device", className='tab-block', children=[
                            html.H3('Computing Devices'),
                            html.Div([
                                        "Select Devices",
                                        dcc.Checklist(id="select-devices",
                                            options=[],
                                            value=['cpu'],
                                            labelStyle={'display':'block!important'}
                                        )
                            ])
                        ], style={'display':'inline-block', 'width':'20%'}),
                    ], style={  'min-height':'400px'
                                ,'display':'flex'
                                ,'display':'-webkit-box'
                                ,'display':'-webkit-flex'
                                ,'-webkit-flex-diredtion':'row'
                                ,'flex-direction':'row'
                                ,'align-items':'stretch'}),
                html.Div(id = "block-file", className='tab-block', children=[
                    html.Div(className= 'file-upload-div', children=[
                        html.H3('Dataset Selection'),
                        dcc.Upload(id = 'file-ul-req', children=[
                            html.Div(['Drag and Drop or Select File'
                            ]),
                        ], style ={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',                            
                            },
                            multiple=False,
                            max_size=-1
                        ),
                        dcc.Loading( children = [
                        html.Div(id='display-ul-data', children=[
                            #dcc. add table para que no haya errores al inicio
                            dash.dash_table.DataTable(id='dataset-table')
                        ])]),
                        html.Div(id='bacthsize-div', children=[
                            html.H5('Batchsize:'),
                            dcc.Input(id='batchsize-input'
                                     ,type='number'
                                     ,placeholder='Enter the batchsize...'
                                     ,min=1
                                     ,value=16)
                        ])
                    ])
                ]),
               
                dcc.Loading(children = [
                html.Div(id='saved-ses-sink', children = [
                    dcc.Loading( children=[
                        html.Div( className='tab-block', children=[
                            html.H3('Load Saved Session'),
                            dcc.Upload(id='load-saved-session', children = [
                                'Drag and Drop or Select File'
                            ]
                             ,style ={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px',                            
                            },
                            multiple=False,
                            max_size=-1
                            )
                        ])
                    ])
                ])]),
                 html.Div( children=[
                    html.Button('Add Request', id="add-request"
                                             , n_clicks=0
                                             , disabled=True
                                             , style={  'width' : '100%',
                                                        'position':'fixed',
                                                        'bottom':0,
                                                        'margin-left':'-50%'})
                    ],
                    style={'text-align':'center'}),
            ]),
        ])
            
    ])
]) 

if __name__=='__main__':
    print('Client startup...')

    #If there is a server passed as an argument, else defaults to localhost:42001
    #in callbacks_req module

    if len(sys.argv) > 1:
        callbacks_req.request_server = callbacks_req.grpc_if_start(sys.argv[1])
    
    wsgi_port = 42000 # <--- get parameter from electronjs main
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        while not sock.connect_ex(('localhost', wsgi_port)):
            #sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            wsgi_port+=1
    except Exception as e:
        print(e)
    finally:
        sock.close()

    print(f'Client internal port: {wsgi_port}')
    serve(app.server, host='localhost', port=wsgi_port)
    
    #TEST CASE
    """    import grpc_if_methods as g
        test= g.Server_grpc_if("localhost:42001")
        print(test.downloadModel('bert-base-uncased'))
        print(test.downloadModel('bert-base-cased'))

        print(test.getModels())
        print(test.deleteModel('bert-base-uncased'))
        print(test.deleteModel('bert-base-cased')) """

