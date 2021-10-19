import sys, getopt
from pathlib import Path
from dash_core_components.Loading import Loading

from pandas.io.formats import style
sys.path.append(str(Path(__file__).parents[2]))

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq

from app import app
import callbacks_req
import callbacks_inf

from waitress import serve


app.title = "AllSpark"
app.layout = html.Div(id='app-index',children=[
    dcc.Store(id='tab-count', storage_type='session', data=1),
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
    ),
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
                                                {'label': 'CLS', 'value': 'cls'},
                                                {'label': 'Sum', 'value': 'sum'},
                                                {'label': 'Average', 'value': 'avg'},
                                                {'label': 'F Joint', 'value': 'f_joint'},
                                                {'label': 'F Ind', 'value': 'f_ind'},
                                                {'label': 'F Inf', 'value': 'f_inf'},
                                                {'label': 'Sentence Transformers (Siamese)', 
                                                        'value': 'siamese', 
                                                        'disabled':True},
                                            ],
                                            value='cls',
                                            labelStyle={'display':'block!important'}
                                        )
                            ])
                        ], style={'display':'inline-block', 'width':'20%'}),
                        html.Div(id = "block-model", className='tab-block', children=[
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
                                                
                                            ) ]) ,  ], style={"display": "inline-block","position":"relative"})
                                    ]),
                                    html.Div([
                                        "Select Model",
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
                            ])   
                            
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
                    ], style={'display':'flex'
                                ,'display':'-webkit-box'
                                ,'display':'-webkit-flex'
                                ,'-webkit-flex-diredtion':'row'
                                ,'flex-direction':'row'
                                ,'align-items':'stretch'}),
                html.Div(id = "block-file", className='tab-block', children=[
                    html.Div(className= 'file-upload-div', children=[
                        html.H3('Dataset Selection'),
                        dcc.Upload(id = 'file-ul-req', children=[
                            html.Div(['Drag and Drop or Select File(s)'
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
                html.Div(children=[
                    html.Button('Add Request', id="add-request", n_clicks=0, disabled=True)
                ])
            ]),
        ])
            
    ])
]) 

if __name__=='__main__':
    print('Client startup...')


    serve(app.server, host='localhost', port=42000)
    """    import grpc_if_methods as g
        test= g.Server_grpc_if("localhost:42001")
        print(test.downloadModel('bert-base-uncased'))
        print(test.downloadModel('bert-base-cased'))

        print(test.getModels())
        print(test.deleteModel('bert-base-uncased'))
        print(test.deleteModel('bert-base-cased')) """

