import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import pandas as pd
import umap

class dim_reduct():
    def __init__(self):
        pass
    def pca(self, embeddings, x=0, y=1, z=2, dim_red = 10):
        output = embeddings
        min_size = min([len(embeddings), len(embeddings[0])])
        if not min_size<3:
            i= min([min_size, dim_red])  #we make sure PCA is not performed on more dimensions than needed
            pca_i=PCA(n_components=i)
            output = pca_i.fit_transform(embeddings)
            output = output[:,[x,y,z]]
        return output

    def tsne(self, embeddings, perplexity=30, it=250, learning_r=200): #per 5-50 learning 2-300
        output = []
        min_size = min([len(embeddings), len(embeddings[0])])
        
        """ if (len(embeddings[0]>75)) and not min_size<3:
            pca_i = PCA(n_components=50)
            embeddings = pca_i.fit_transform(embeddings) """
       
        if not min_size<2:
            model = TSNE(n_components=3
                        ,random_state=0
                        ,perplexity=perplexity
                        ,n_iter=it
                        ,learning_rate=learning_r
                        #,init='pca'
                        ,n_jobs=-1)
            output = model.fit_transform(embeddings)
        return output

    def umap(self, embeddings, neighb=15):
        output = embeddings
        if not len(embeddings)<5:
            umap_i = umap.UMAP(n_components=3, n_neighbors=neighb)
            output = umap_i.fit_transform(embeddings)
        return output
    
    def graph_run(self, components, sentences):
        components = pd.DataFrame(components)
        components['sentence'] = sentences
        fig = px.scatter_3d(components, x=0, y=1, z=2, template='plotly_dark', hover_data=['sentence'])
        fig.update_traces(marker=dict(size=4, line=dict(width=1, color='DarkSlateGrey')))
        
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
                            modebar_orientation='v'
                        )
        return fig
        #fig.show()

class sent_simil():
    def __init__(self):
        pass
    def cosine(self):
        pass
    def icmb(self):
        pass





if __name__=='__main__':
    vectors = np.random.rand(500, 768)

    a = [f'{i}' for i in range(500)]

    print(vectors)
    #vectors = np.array([range(100) for _ in list(range(100))])
    reductor = dim_reduct()
    resultado =  reductor.pca(vectors, 0,1,2)
    import pandas as pd

    resultado = pd.DataFrame(resultado)
    a = pd.DataFrame(a)
    a.columns = ['sentences']
    #print(a)
    resultado['sentences'] = a
    #resultado = reductor.tsne(vectors, 30, 250, 200)
    #resultado= reductor.umap(vectors, 15)
    reductor.graph_run(resultado, a) 
   
   