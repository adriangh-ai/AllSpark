from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from datetime import datetime
import os

import numpy as np

from dash.dependencies import ALL, Input, MATCH, Output, State
import dash
from dash import dcc
from dash import html
from dash import dash_table

import pandas
import plotly.express as px
from torch import layout

from app import app

from representation import dim_reduct, sent_simil

WORKDIR = Path(__file__).parent     # Program base file tree
HOME = Path.home()
SAVED_FOLDER = Path.joinpath(HOME, 'AllSpark_data/')

request_record = {}                 # Record where the requests are temporary stored
tab_record = OrderedDict()          # Record where the Inference Tabs are stored

class Inference():
    """
    Single instance of inference element.

    Params:
        index(int):         index in the tab dictionary
        model(string):      name of the model
        dataset(DataFrame): original dataset with the sentences
        filename(string):   name of the dataset file
        text_column(string):selected column for the sentences
        layer_low(int):     lower model layer
        layer_up(int):      upper model layer
        comp_func(string):  name of the chosen composition function
        batchsize(int):     batchsize for inference
        devices(list)       list of chosen devices for the operations
        embeddings(nArray): output embeddings from the inference process
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
                    ,devices
                    ,embeddings = []):

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
            self.embeddings = embeddings
            self.adyacency = self._calc_adjacent()
        
    def set_embeddings(self, embed_dict):
        """
        Sets the sentence embeddings given by the server as dictionary, converts them
        to numpy array.

        Args:
            embed_dict: dictionary containing the embeddings
        """
        embed_stack = []
        for sentence in embed_dict:
            embed_stack.append(np.array(sentence['value'], dtype=np.float32))
        self.embeddings = np.stack(embed_stack)

    def get_dataset(self):
        """
        Getter for dataset.
        """
        #return self.dataset[self.text_column].squeeze().to_list()
        return self.dataset[self.text_column]
    
    def get_sentence_data(self):
        """
        Returns unique sentences and their original column
        """
        return [[sentence, self.text_column] for sentence in self.adyacency.keys()]
    
    def _calc_adjacent(self):
        _adyacents = defaultdict(list)
        for idx, k in enumerate(self.dataset[self.text_column].tolist()):
            _adyacents[k].append(idx)
        return _adyacents

class inference_tab():
    """
    This class contains the details of the Inference Tab it represents,
    identified by the index argument.

    Params:
        index(int):         index in the tab dictionary
        model(string):      name of the model
        dataset(DataFrame): original dataset with the sentences
        filename(string):   name of the dataset file
        text_column(string):selected column for the sentences
        layer_low(int):     lower model layer
        layer_up(int):      upper model layer
        comp_func(string):  name of the chosen composition function
        batchsize(int):     batchsize for inference
        devices(list)       list of chosen devices for the operations
        embeddings(nArray): output embeddings from the inference process
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
                ,devices
                ,embeddings = []):

        self.index=index
        self.model=model
        #self.dataset = dataset
        self.filename= filename
        self.text_column=text_column
        self.layer_low = layer_low
        self.layer_up = layer_up
        self.comp_func = comp_func
        self.batchsize = batchsize
        self.devices = devices
        #self.embeddings = embeddings
        
        self.dimred_funct = dim_reduct()
        self.simil = sent_simil()

        self.inference_list = [Inference(index
                                        ,model
                                        ,dataset[column].to_frame()
                                        ,filename
                                        ,column
                                        ,layer_low
                                        ,layer_up
                                        ,comp_func
                                        ,batchsize
                                        ,devices
                                        ,embeddings) for column in text_column]
        
        self.colors=['blue','red','green', 'yellow', 'white'] 
    
    def __getitem__(self, index): 
        try:
            return self.inference_list[index]
        except IndexError as e:
            return 'Index Error'
    
    def append(self, inf_elem):
        """
        Adds an inference object to the wrapper's list

        Args:
            inf_element: Inference object
        """
        self.inference_list.append(inf_elem)

    def set_embeddings(self, embed_dict):
        """
        Sets the sentence embeddings given by the server as dictionary, to each inference object
        to handle.

        Args:
            embed_dict: dictionary containing the embeddings
        """
        idx = 0
        for element in self.inference_list:
            element.set_embeddings(embed_dict[idx:(idx+len(element.adyacency))])
            idx+=len(element.adyacency)

    def get_sentence_list(self):
        """
        Getter for dataset.
        """
        #return self.dataset[self.text_column].squeeze().to_list()
        _sentence_list = []
        for i in self.inference_list:
            _sentence_list.extend(i.adyacency.keys())
        return _sentence_list
    
    def get_sentence_data_list(self):
        """
        Getter for dataset.
        """
        #return self.dataset[self.text_column].squeeze().to_list()
        _sentence_list = []
        for i in self.inference_list:
            _sentence_list.extend(i.get_sentence_data())
        return _sentence_list
    
    def generate_fig(self, tab_val, params):
        """
        Generates the Plotly figure object with the selected dimensionality reduction
        method.

            Args:
                tab_val: String with the dimensionality reduction
                params: List with the parameters needed by the reduction function
            Return:
                Plotly figure object
        """

        points = np.concatenate([i.embeddings for i in self.inference_list])

        if tab_val == 'pca-tab':
            points = self.dimred_funct.pca(points, *params)
        elif tab_val == 'tsne-tab':
            points = self.dimred_funct.tsne(points,*params)
        elif tab_val == 'umap-tab':
            points = self.dimred_funct.umap(points, *params)

        # Wrapper Plotly Figure
        fig = px.scatter_3d(template='plotly_dark')
        
        fig.update_traces(marker=dict(size=3, line=dict(width=1, color='DarkSlateGrey')))
        
        fig.update_traces(hoverinfo="text",selector=dict(type='scatter3d'))

        fig.update_layout(scene = dict(
                            xaxis = dict(
                                backgroundcolor="rgb(200, 200, 230)",
                                gridcolor="grey",
                                showbackground=False,
                                zerolinecolor="white", zerolinewidth = 5, ),
                            yaxis = dict(
                                backgroundcolor="rgb(230, 200,230)",
                                gridcolor="grey",
                                showbackground=False,
                                zerolinecolor="white", zerolinewidth = 5),
                            zaxis = dict(
                                backgroundcolor="rgb(230, 230,200)",
                                gridcolor="grey",
                                showbackground=False,
                                zerolinecolor="white", zerolinewidth=5),),
                            margin=dict(
                            r=10, l=10,
                            b=10, t=10),
                            #modebar_orientation='v'
                        )
        #fig.update_layout(clickmode='event+select')

        fig.update_layout(legend=dict(
                            x=0,
                            y=0.5,
                            xanchor='left',
                            yanchor='middle',
                            title_font_family="Times New Roman",
                            font=dict(
                                family="Courier",
                                size=12,
                                color="gray"
                            ),
                            bordercolor="gray",
                            borderwidth=2
                        )
        )

       # Recolor diferent traces and add them into the wrapper
        indx = 0
        for idx, elem in enumerate(self.inference_list):
                _elem_trace = self.dimred_funct.graph_run(
                                                points[indx:(indx+len(elem.adyacency))]
                                                ,elem.adyacency.keys()
                )
                
                _elem_trace.update_traces(
                                            showlegend=True
                                            ,name=elem.text_column
                                            ,marker=dict(color=self.colors[len(self.colors)%(idx+1)])
                )

                fig.add_traces(data=_elem_trace.data)
                indx+=len(elem.adyacency)
        import plotly.graph_objects as go
    #figure['data'].append(go.Scatter3d(x=[1,2,3], y = [1,2,3], z=[1,2,3], line=dict(color='yellow')))
        fig.add_traces(go.Scatter3d(line=dict(color='yellow')))
        fig.update_layout(uirevision=True)
        return fig

    def generate_tab(self):
        """
        Generates the tab layout given the class attributes after the inference process.
        Return:
            Dash Object
        """
        # Guard for Inference Errors
        for inf_elem in self.inference_list:
            if (1,1,1) in inf_elem.embeddings:
                new_tab = dcc.Tab(label = f'Inference {self.index}'
                                ,id={'type':'inf-tab', 'index': self.index }
                                ,className='main-tabs', children=[
                                    html.Div(
                                        html.P('There was an error processing the data. Please be sure to check that the model and parameters are correct.',
                                        style={'text-align':'center'})
                                    )
                                ])
                return new_tab
        # Page display
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
                                    html.P("Sentence Similarity"),
                                    dcc.RadioItems(id={'type':'similarity-radio', 'index': self.index },
                                        options=[
                                            {'label':'No Similarity', 'value': False},
                                            {'label': 'Cosine', 'value':'cos'},
                                            {'label':'ICMB', 'value':'icmb'}
                                        ],
                                        value=False
                                    ),
                                    html.Div( children =[
                                        html.P("B Selector"),
                                        dcc.Slider(id={'type':'b-slider', 'index': self.index },
                                            value = 1,
                                            min=1,
                                            max=2,
                                            step=0.1,
                                            marks = {i:f'{round(i,2)}' for i in np.arange(1,2,0.1)}
                                        )
                                        ]),

                                    html.Button('Save Session', id= {'type':'save-session'
                                                                    ,'index': self.index },
                                                                    style={'position':'fixed','bottom':200})
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
                                    'displayModeBar':True,
                                    'responsive': True

                                }
                                )],
                        style={'display':'inline-block'}) ]),
                        
                        ], style={'display':'flex'
                                ,'display':'-webkit-box'
                                ,'display':'-webkit-flex'
                                ,'-webkit-flex-diredtion':'row'
                                ,'flex-direction':'row'
                                ,'align-items':'stretch'}),
                        html.Div(id = {'type':'simil-table-div', 'index': self.index }, children=[
                                    dash_table.DataTable(
                                        id = {'type':'simil-table', 'index': self.index },
                                        data = [],
                                        columns = [{'name':'Sentence Similarity', 'id':'Sentence Similarity'}], 
                                        style_table={'overflowY': 'scroll'},
                                        style_cell={'textAlign':'left'},
                                        column_selectable='single'
                                    ),
                            ],
                            style={'position':'fixed',
                                    'bottom':0,
                                    'width': '100%'}
                        )
            ])
        return new_tab

#######################
#### GRAPH DRAWING ####
#######################

#### Re-Plot graph
@app.callback(
    #Output({'type':'plot-graph', 'index':MATCH}, 'figure'),
    Output({'type':'loading-graph', 'index': MATCH}, 'children'),
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
    """
    Plots the graph, with the dimensionality reduction chosen by the tab selection, using the page
    information given by the user with the chosen parameters.
    """
    if not any([tab_val,pca1,pca2,pca3,tsneper,tsnelearn,tsneiter,uneigh]):
        raise dash.exceptions.PreventUpdate
    
    indx = id['index']
    inf_tab = tab_record.get(indx, [])

    if not inf_tab: # Guard
        return []
    
    params = []
    if tab_val == 'pca-tab':
        params = [pca1, pca2, pca3]
    elif tab_val == 'tsne-tab':
        params = [tsneper, tsneiter, tsnelearn]
    elif tab_val == 'umap-tab':
        params = [uneigh]

    return [dcc.Graph(id={'type':'plot-graph', 'index': inf_tab.index },
                                style={'height':'100%'},
                                responsive=True,
                                figure=inf_tab.generate_fig(tab_val, params),
                                config= {
                                    'autosizable':True,
                                    'displaylogo':False,
                                    'fillFrame':True,
                                    'modeBarButtons':'hover',
                                    'displayModeBar':True,
                                    'responsive': True

                                }
                                )]
########################
#### Draw relations ####
########################
""" @app.callback(
    Output({'type':'plot-graph', 'index':MATCH}, 'figure'),
    Input({'type':'plot-graph', 'index':MATCH}, 'clickData'),
    State({'type':'plot-graph', 'index':MATCH}, 'figure'),
    State({'type':'plot-graph', 'index':MATCH}, 'id')
)
def draw_point_relations(clickdata, figure, id):
    indx = id['index']
    inf_tab = tab_record.get(indx, [])
    
    if not inf_tab:
        return []
    
    if not inf_tab or not figure or not clickdata: 
        raise dash.exceptions.PreventUpdate
    
    #Delete previous traces
    figure['data'] = figure['data'][:len(inf_tab[:])]
    #Draw new trace
    import plotly.graph_objects as go
    figure['data'].append(go.Scatter3d(x=[1,2,3], y = [1,2,3], z=[1,2,3], line=dict(color='yellow')))
    print(figure.get('layout','nope!'))
 
    return figure """
###################
#### Similarity####
###################

@app.callback(
    Output({'type':'simil-table-div', 'index':MATCH}, 'children'),
    Input({'type':'plot-graph', 'index':MATCH}, 'clickData'),
    Input({'type':'similarity-radio', 'index': MATCH }, 'value'),
    Input({'type':'b-slider', 'index': MATCH }, 'value'),
    State({'type':'plot-graph', 'index':MATCH}, 'selectedData'),
    State({'type':'plot-graph', 'index':MATCH}, 'id'),
    State({'type':'plot-graph', 'index':MATCH}, 'figure')
)
def similarity_table(clickdata,valuesim, valueb ,selected,id,fig):
    """
    Takes the selected point from the graph, then perfroms the selected similarity function
    over caller tab embeddings.
    """
    indx = id['index']
    inf_tab = tab_record.get(indx, [])
    
    if not inf_tab:
        return []
    
    if not inf_tab: 
        raise dash.exceptions.PreventUpdate
    
    if not valuesim:
        return []
    
    sorted_sentences = pandas.DataFrame({'A' : []})
    if clickdata:
        print(clickdata)
        # Isolate point and trace from figure
        points = inf_tab[ clickdata['points'][0]['curveNumber'] ].embeddings
        vector1 = points[ clickdata['points'][0]['pointNumber'] ]
        points = np.concatenate([i.embeddings for i in inf_tab[:]])

        # Similarity function calls
        simil_array = []
        simil_method = 'distance'
        if 'cos' in valuesim:
            simil_array = inf_tab.simil.cosine(vector1, points)
            simil_method = 'cosine'
        else:
            simil_array = inf_tab.simil.icmb(vector1, points, valueb)
        
        # Sorting and formatting
        simil_array = inf_tab.simil.sorted_simil(simil_array,simil_method)[:10]
        _sentences = inf_tab.get_sentence_data_list()
       
        _dataframe_list = [_sentences[s_idx] for s_idx in simil_array]
        sorted_sentences = pandas.DataFrame(_dataframe_list , columns={'Sentence Similarity','Dataset'})
        sorted_sentences = sorted_sentences[['Sentence Similarity', 'Dataset']]
    # Dash repr
    sorted_sentences = sorted_sentences.head(10).to_dict('records') if not sorted_sentences.empty else []
    _children = [dash_table.DataTable(
                    id = {'type':'simil-table', 'index': indx },
                    data = sorted_sentences,
                    columns = [{'name':"Dataset", 'id':'Dataset'},
                                {'name':"Sentence Similarity", 'id':'Sentence Similarity'}], 
                    style_table={'overflowY': 'scroll'},
                    style_cell={'textAlign':'left'},
                    column_selectable='single',
                    style_data_conditional= [{
                        'if' : {
                            'row_index' : 0
                        },              
                        'color':'#636efa'
                    }]
                )]
    return _children


######################
#### SAVE TO FILE ####
######################

@app.callback(
    Output({'type':'save-session','index' : MATCH}, 'value'),
    Input({'type':'save-session','index' : MATCH}, 'n_clicks'),
    State({'type':'save-session','index' : MATCH}, 'id')
)
def save_session(n_clicks, id):
    """
    Save the sessions results (embeddings, modelname, filename, dataset column, chosen layers and
    composition method) to file.
    """
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    index = id['index']
    inf_tab = tab_record[index]
    
    ################
    ## haz excepciones antes de tocar, crea el folder, etc
    _saved_dict = {}
    for i in inf_tab[:]:
        data_session = i.get_dataset().to_frame()
        data_session.rename({i.text_column : 'sentences'}, axis=1, inplace = True)
        data_session = data_session.to_dict('list')

        data_session['value'] = i.embeddings.tolist()
        data_session['model'] = i.model
        data_session['filename'] = i.filename
        data_session['column'] = i.text_column
        data_session['layers'] = [i.layer_low, inf_tab.layer_up]
        data_session['composition'] = i.comp_func
        _saved_dict[i.text_column] = data_session
    
    _timestamp = datetime.now().strftime("%m_%d_%Y_%H%M%S")
    saved_file = Path.joinpath(SAVED_FOLDER, f"{inf_tab[0].model}{_timestamp}.aspk") #first element model
    
    if not os.path.exists(SAVED_FOLDER): #Creates folder if it doesn't exist
        os.makedirs(SAVED_FOLDER)

    with open(saved_file, 'w+') as file:
        json.dump(_saved_dict, file)
        
    return 0

if __name__ == '__main__':
    a = inference_tab('test','test','test','test','test',1,2,'fc',13,[1,2])
    print(a[0].model)
    for i in a[:]:
        print('test1')