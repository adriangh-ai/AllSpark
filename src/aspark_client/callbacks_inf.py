from json import load
from dash.dependencies import ALL, Input, MATCH, Output, State
import dash
import dash_core_components as dcc
from dash_core_components.RadioItems import RadioItems
import dash_html_components as html

from collections import OrderedDict
from dash_html_components.Label import Label
import numpy as np

from representation import dim_reduct, sent_simil
import time

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
        self.dimred_funct = dim_reduct()
    
    def set_embeddings(self, embed_dict):
        embed_stack = []
        for sentence in embed_dict:
            embed_stack.append(np.array(sentence['value'], dtype=np.float32))
        self.embeddings = np.stack(embed_stack)
    def get_sentence_list(self):
        #return self.dataset[self.text_column].squeeze().to_list()
        return self.dataset[self.text_column]
    
    def generate_tab(self):
        new_tab = dcc.Tab(label = f'Inference {self.index}'
                            ,id={'type':'inf-tab', 'index': self.index }
                            ,className='main-tabs', children=[
                    html.Div(  children=[
                        html.Div( children =[
                            html.Div( children=[
                                dcc.Tabs( id={'type':'dim-red-tabs', 'index': self.index}, 
                                className='main-tabs',
                                children=[
                                    dcc.Tab(label='PCA',id={'type':'PCA-tab', 'index': self.index }, 
                                    className='main-tabs',
                                    children=[
                                        html.P('Select components: sorted by "explained variance"'),
                                        html.Div(  children=[
                                            html.H5('Component 1:'),
                                            dcc.Dropdown(id={'type':'pca-dim1-dd', 'index': self.index },
                                                options=[
                                                    {'label':f'#{i}.', 'value':i} for i in list(range(1,11))
                                                ],
                                                value=1
                                            )
                                        ]),
                                        html.Div(  children=[
                                            html.H5('Component 2:'),
                                            dcc.Dropdown(id={'type':'pca-dim2-dd', 'index': self.index },
                                                options=[
                                                    {'label':f'#{i}.', 'value':i} for i in list(range(1,11))
                                                ],
                                                value=2
                                            )
                                        ]),
                                        html.Div(  children=[
                                            html.H5('Component 3:'),
                                            dcc.Dropdown(id={'type':'pca-dim3-dd', 'index': self.index },
                                                options=[
                                                    {'label':f'#{i}.', 'value':i} for i in list(range(1,11))
                                                ],
                                                value=3
                                            )
                                        ]),
                                    ], value='pca-tab'),
                                    dcc.Tab(label='T-SNE', 
                                    className='main-tabs',
                                    children=[
                                        html.Div( children =[
                                        html.P("Perplexity:"),
                                        dcc.Slider(id={'type':'tsne-perp-slider', 'index': self.index },
                                            value=30,
                                            min = 5,
                                            max = 50,
                                            marks = {i:f'{i}' for i in range(5,51,5)}
                                        )
                                        ]),
                                        html.Div( children =[
                                        html.P("Learning rate:"),
                                        dcc.Slider(id={'type':'tsne-learnrate-slider', 'index': self.index },
                                            value=200,
                                            min=10,
                                            max=1000,
                                            step=10,
                                            marks = {i:f'{i}' for i in range(10,1001,100)}
                                        )
                                        ]),
                                        html.Div( children =[
                                        html.P("Iterations:"),
                                        dcc.Slider(id={'type':'tsne-iter-slider', 'index': self.index },
                                            value = 300,
                                            min=1,
                                            max=5000,
                                            step=50,
                                            marks = {i:f'{i}' for i in range(1,5001,1002)}
                                        )
                                        ])
                                    ], value='tsne-tab'),
                                    dcc.Tab(label='UMAP', 
                                    className='main-tabs',
                                    children=[
                                        html.Div( children =[
                                        html.P('Neighbors:'),    
                                        dcc.Slider(id={'type':'umap-neighb-slider', 'index': self.index },
                                            value=15,
                                            min=2,
                                            max=200,
                                            marks = {i:f'{i}' for i in range(2,201,42)}
                                        )
                                        ])
                                    ], value='umap-tab')
                                ]),
                                html.Button('RE-PLOT',id={'type':'replot-button', 'index': self.index })
                            ]),
                                html.Div( children=[
                                    dcc.RadioItems(id={'type':'similarity-radio', 'index': self.index },
                                        options=[
                                            {'label': 'Cosine', 'value':'cos'},
                                            {'label':'ICMB', 'value':'icmb'}
                                        ]
                                    ),
                                    html.Ul( children=[
                                        
                                    ]),
                                    html.Button('Close Tab', id= {'type':'close-tab'
                                                                    ,'index': self.index })
                                    ])
                            ],style={'display':'inline-block','width':'300px', 'min-width':'300px'}),
                        
                        html.Div( id = {'type' : 'graph-wrapper', 'index':self.index }, children = [
                            dcc.Loading(id={'type':'loading-graph', 'index': self.index}, children = [
                            dcc.Graph(id={'type':'plot-graph', 'index': self.index },
                                style={'height':'100%'},
                                responsive=True,
                                config= {
                                    'autosizable':True,
                                    'displaylogo':False,
                                    'fillFrame':True,
                                    'modeBarButtons':'hover',
                                    'displayModeBar':'hover'

                                }
                                )],
                        style={'display':'inline-block'}) ]),
                        
                        ], style={'display':'flex'
                                ,'display':'-webkit-box'
                                ,'display':'-webkit-flex'
                                ,'-webkit-flex-diredtion':'row'
                                ,'flex-direction':'row'
                                ,'align-items':'stretch'}),
                        
            ])
        return new_tab

@app.callback(
    Output({'type':'plot-graph', 'index':MATCH}, 'figure'),
    Input({'type':'replot-button', 'index':MATCH}, 'n_clicks'),
    Input({'type':'dim-red-tabs', 'index': MATCH}, 'value'),
    State({'type':'pca-dim1-dd', 'index': MATCH}, 'value'),
    State({'type':'pca-dim2-dd', 'index': MATCH}, 'value'),
    State({'type':'pca-dim3-dd', 'index': MATCH}, 'value'),
    State({'type':'tsne-perp-slider', 'index': MATCH}, 'value'),
    State({'type':'tsne-learnrate-slider', 'index': MATCH}, 'value'),
    State({'type':'tsne-iter-slider', 'index': MATCH}, 'value'),
    State({'type':'umap-neighb-slider', 'index': MATCH}, 'value'),
    State({'type':'plot-graph', 'index':MATCH}, 'id')
)
def figure_update(n_clicks, tab_val, pca1, pca2, pca3, tsneper, tsnelearn, tsneiter, uneigh, id):
    indx = id['index']
    inf_tab = tab_record[indx]
    points = inf_tab.embeddings
    print(len(points))
    if not tab_val:
        raise dash.exceptions.PreventUpdate
    print(tab_val)
    if tab_val == 'pca-tab':
        points = inf_tab.dimred_funct.pca(inf_tab.embeddings, pca1-1, pca2-1, pca3-1)
    elif tab_val == 'tsne-tab':
        points = inf_tab.dimred_funct.tsne(inf_tab.embeddings,tsneper, tsneiter, tsnelearn)
    elif tab_val == 'umap-tab':
        points = inf_tab.dimred_funct.umap(inf_tab.embeddings, uneigh)
    
    return inf_tab.dimred_funct.graph_run(points, inf_tab.get_sentence_list())

""" @app.callback(
    Output({'type':'dim-red-loading-output', 'index': MATCH}, 'loading-state'),
    Input({'type':'dim-red-tabs', 'index': MATCH }, 'value'),
    Input({'type':'graph-wrapper', 'index': MATCH }, 'children')
)
def graph_loading(value, value2):
    print(app.title)
    time.sleep(1)
    return value2 """

""" @app.callback(
    Output({'type':'main-tab', 'index':MATCH}, 'children'),
    Input({'type':'close-tab','index':MATCH}, 'n_clicks'),
    State({'type':'close-tab','index':MATCH }, 'id'),
    State('main-tab', 'children')
)
def close_tab(n_clicks, id, children):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    indx = id['index']
    tab_keys = list(tab_record.keys())
    del_index = tab_keys.index(indx)
    del tab_record[indx]
    children.pop(del_index)
    return children """