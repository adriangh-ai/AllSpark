import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
import umap

class dim_reduct():
    def __init__(self):
        pass
    def pca(self, embeddings, x=0, y=1, z=2):
        output = embeddings
        if not len(embeddings)<3:
            i= min([len(embeddings), 10])  #we make sure PCA is not performed on more dimensions than needed
            pca_i=PCA(n_components=i)
            output = pca_i.fit_transform(embeddings)
            output = output[:,[x,y,z]]
        return output

    def tsne(self, embeddings, perplexity=30, it=250, learning_r=200): #per 5-50 learning 2-300
        output = embeddings
        if not len(embeddings)<2:
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

class sent_simil():
    def __init__(self):
        pass
    def cosine(self):
        pass
    def icmb(self):
        pass

def graph_run(components):
    fig = px.scatter_3d(components, x=0, y=1, z=2, template='plotly_dark')
    fig.update_traces(marker=dict(size=3))
    #total_var = pca.explained_variance_ratio_.sum() * 100
    #print(total_var)
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
                        b=10, t=10)
                    )
    return fig
    #fig.show()



if __name__=='__main__':
    vectors = np.random.rand(1000, 768)
    #vectors = np.array([range(100) for _ in list(range(100))])
    reductor = dim_reduct()
    #resultado =  reductor.pca(vectors, 0,1,2)
    #resultado = reductor.tsne(vectors, 30, 250, 1)
    resultado= reductor.umap(vectors, 15)
    graph_run(resultado)