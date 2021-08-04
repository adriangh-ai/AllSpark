import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import grpc
from src.grpc_files import compservice_pb2, compservice_pb2_grpc

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State

from app import app
import callbacks_req
import callbacks_inf
#from layouts import layout1, layout2

from waitress import serve
app.title = "AllSpark"
app.layout = html.Div([
    html.Div(
        className="app-index-header",
        children=[
            html.Div(className="app-header--title", children=[
                html.H1("AllSpark", title="AllSpark")
            ])
        ]
    ),
    html.Div(
        children = [dcc.Tabs(id="main-tab",className="main-tabs", value='tab-1', children=[
            dcc.Tab(label='Session', value='tab-1', className='main-tabs', children=[
                html.Div(id = "block-req-list", className='tab-block', children=[
                    "aquí es donde se pone la lista de requestst"
                ]),
                html.Div(id = "block-model", className='tab-block', children=[
                    html.H3('Model Selection', style={"align":"centered"}),
                    html.Div([ html.Label('HugginFace Model Repository'),
                            html.Div([
                                dcc.Dropdown(id='model-dropdown', className="dropdown-menu", options=[],
                                    multi=False,
                                    style={"min-width":"75%", "display":"inline-block"}
                                ),
                                html.Button(children=["Download"],
                                    style={"display": "inline-block","position":"relative"}
                                )
                            ]),
                            html.Div([
                                "Select Model",
                                dcc.RadioItems(
                                    options=[
                                        {'label': 'New York City', 'value': 'NYC'},
                                        {'label': 'Montréal', 'value': 'MTL'},
                                        {'label': 'San Francisco', 'value': 'SF'}
                                    ],
                                    value='MTL',
                                    labelStyle={'display':'block!important'}
                                ),
                                html.Button(children=["Delete"],
                                    style={"display": "inline-block","position":"relative"}
                                )  
                            ]),
                            
                            html.Div(id="layer-div", children=[
                                html.Label('Layer selector'),
                                dcc.RangeSlider(
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
                    
                ]),
                html.Div(id = "block-file", className='tab-block', children=[
                    html.P('this is where the file is added')
                ]),
                html.Div(id = "block-composition", className='tab-block', children=[
                    html.Div([
                                "Choose Composition Method",
                                dcc.RadioItems(
                                    options=[
                                        {'label': 'New York City', 'value': 'NYC'},
                                        {'label': 'Montréal', 'value': 'MTL'},
                                        {'label': 'San Francisco', 'value': 'SF'}
                                    ],
                                    value='MTL',
                                    labelStyle={'display':'block!important'}
                                )
                    ])
                ]),
                html.Div(id = "block-device", className='tab-block', children=[
                    html.Div([
                                "Select Devices",
                                dcc.Checklist(
                                    options=[
                                        {'label': 'New York City', 'value': 'NYC'},
                                        {'label': 'Montréal', 'value': 'MTL'},
                                        {'label': 'San Francisco', 'value': 'SF'}
                                    ],
                                    value='MTL',
                                    labelStyle={'display':'block!important'}
                                )
                    ])
                ]),
                html.Div(children=[
                    html.Button('Add Request', id="add-tab", n_clicks=0, name="nonse")
                ])
            ])
        ])
            
    ])
]) 


""" @app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/apps/app1':
         return layout1
    elif pathname == '/apps/app2':
         return layout2
    else:
        return '404'
 """

if __name__=='__main__':
    print('wtf')
    print(callbacks_req.stub.getDevices(compservice_pb2.Dummy(dummy3=1)))
    serve(app.server, host='localhost', port=42000)