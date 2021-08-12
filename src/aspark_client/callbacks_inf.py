from dash.dependencies import Input, MATCH, Output, State
import dash
import dash_core_components as dcc
from dash_core_components.RadioItems import RadioItems
import dash_html_components as html

from collections import OrderedDict
from dash_html_components.Label import Label
import numpy as np

from representation import dim_reduct, sent_simil

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
    
    def set_embeddings(self, embed_dict):
        embed_stack = []
        for sentence in embed_dict:
            embed_stack.append(np.array(sentence['value'], dtype=np.float32))
        self.embeddings = np.concatenate(embed_stack)
    
    def generate_tab(self):
        new_tab = dcc.Tab(label = f'Inference {self.model}',className='main-tabs', children=[
                    html.Div( id={'type':'inf-tab', 'index': self.index },children=[
                        html.Div( children=[
                            dcc.RadioItems( id={'type':'dim-red-radio', 'index': self.index },
                                options=[
                                    {'label':'PCA','value':'pca'},
                                    {'label':'T-SNE','value':'tsne'},
                                    {'label':'UMAP', 'value':'umap'}
                                ]
                            ),
                            dcc.Tabs( children=[
                                dcc.Tab(label='PCA',id={'type':'PCA-tab', 'index': self.index }, children=[
                                    html.P('Select components: sorted by "explained variance"'),
                                    html.Div(  children=[
                                        html.Label('Component 1:'),
                                        dcc.Dropdown(id={'type':'pca-dim1-dd', 'index': self.index },
                                            options=[
                                                {'label':f'#{i}.', 'value':i} for i in list(range(1,11))
                                            ],
                                            value=1
                                        )
                                    ]),
                                    html.Div(  children=[
                                        html.Label('Component 2:'),
                                        dcc.Dropdown(id={'type':'pca-dim2-dd', 'index': self.index },
                                            options=[
                                                {'label':f'#{i}.', 'value':i} for i in list(range(1,11))
                                            ],
                                            value=2
                                        )
                                    ]),
                                    html.Div(  children=[
                                        html.Label('Component 3:'),
                                        dcc.Dropdown(id={'type':'pca-dim3-dd', 'index': self.index },
                                            options=[
                                                {'label':f'#{i}.', 'value':i} for i in list(range(1,11))
                                            ],
                                            value=3
                                        )
                                    ]),
                                    html.Button('RE-PLOT',id={'type':'pca-replot-button', 'index': self.index },)
                                ]),
                                dcc.Tab(label='T-SNE', children=[
                                    html.Div( children =[
                                    html.P("Perplexity:"),
                                    dcc.Slider(id={'type':'tsne-perp-slider', 'index': self.index },
                                        value=30,
                                        min = 5,
                                        max = 50
                                    )
                                    ]),
                                    html.Div( children =[
                                    html.P("Learning rate:"),
                                    dcc.Slider(id={'type':'tsne-learnrate-slider', 'index': self.index },
                                        value=200,
                                        min=10,
                                        max=1000,
                                        step=10
                                    )
                                    ]),
                                    html.Div( children =[
                                    html.P("Iterations:"),
                                    dcc.Slider(id={'type':'tsne-iter-slider', 'index': self.index },
                                        value = 300,
                                        min=1,
                                        max=5000,
                                        step=50
                                    )
                                    ])
                                ]),
                                dcc.Tab(label='UMAP', children=[
                                    html.Div( children =[
                                    html.P('Neighbors:'),    
                                    dcc.Slider(id={'type':'umap-neighb-slider', 'index': self.index },
                                        value=15,
                                        min=2,
                                        max=200,
                                    )
                                    ])
                                ])
                            ])
                        ],style={'display':'inline-block','width':'300px'}),
                        html.Div(
                            dcc.Graph(id={'type':'plot-graph', 'index': self.index },
                                style={'height':'100%'}
                                ),
                        style={'display':'inline-block','height':'100%', 'min-width':'70%'}),
                        html.Div( children=[
                            dcc.RadioItems(id={'type':'similarity-radio', 'index': self.index },
                                options=[
                                    {'label': 'Cosine', 'value':'cos'},
                                    {'label':'ICMB', 'value':'icmb'}
                                ]
                            ),
                            html.Ul( children=[
                                
                            ]),
                            html.Button('will this work', id= {'type':'close-tab'
                                                             ,'index': self.index }, n_clicks=0)
                            ],style={'display':'inline-block', 'width':'15%'})
                        ], style={'display':'flex'
                                ,'display':'-webkit-box'
                                ,'display':'-webkit-flex'
                                ,'-webkit-flex-diredtion':'row'
                                ,'flex-direction':'row'
                                ,'height':'100%'
                                ,'align-items':'stretch'}),
                        
            ])
        return new_tab