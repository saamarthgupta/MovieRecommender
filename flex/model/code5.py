import gensim.downloader as api
from pyemd import emd
from nltk.corpus import stopwords
from nltk import download
from nltk.cluster import KMeansClusterer
import json
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
from nltk import word_tokenize
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import nltk
import numpy as np
from random import shuffle
from scipy.cluster.hierarchy import dendrogram, linkage
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import logging
import sys
from multiprocessing import cpu_count
from gensim.models.doc2vec import LabeledSentence
import string
import re
RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def preprocess(text):
    '''
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    '''
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
   
    return [stemmer.lemmatize(word) for word in nopunc]
    
def title(text):

    # Remove all punctuation and make all lowercase 
    return RE_PUNCT.sub(" ", text).lower().split()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  

#print(preprocess(title('1917')))
#with open('finalMasterData.json') as json_file:
 #   data = json.load(json_file)
 #   documents=[line['plot'] for line in data]
  #  ids=[line['id'] for line in data]
   # titles=[line['title'] for line in data]
 
#distance = modell.wmdistance(x, y)  # Compute WMD as normal.
#print('distance: %r' % distance)
#finaldocuments = [TaggedDocument(preprocess(documents[i]), ['_'.join(title(titles[i]))]) for i in range(len(titles))]
#print(finaldocuments)
#print(finaldocuments)
#np.savetxt('hi.csv',documents)
#model = Doc2Vec(size=128, window=8, min_count=3,sample=1e-4, workers=3, alpha = 0.025 ,min_alpha = 0.025)
#model.build_vocab(finaldocuments)
fname ='model'

model = Doc2Vec.load(fname)
#doctag = RE_PUNCT.sub(" ",'zindagi na milegi dobara') .lower().split()
#print(model.docvecs.most_similar('_'.join(doctag)))
#print(model.docvecs.most_similar([model[0]]))
#num_epochs = 10
#alpha_delta = (alpha - min_alpha) / num_epochs
alpha=0.018
#for epoch in range(num_epochs):
 #   shuffle(finaldocuments)
  #  model.alpha = alpha
   # model.min_alpha = alpha
   # model.train(finaldocuments,total_examples= len(ids),report_delay=1,epochs=2)
    
   # print(epoch)
   # print("\n")
   # alpha -= 0.0002
#model.init_sims(replace=True)
#model.save(fname)
s="The Godfather"
#p="A tale of a small boy with dreams and his journey to becoming the God of Cricket and the most celebrated sportsperson in his country."
#vec=model.infer_vector(preprocess(p))
doctag = title(s)
#vec = model['adventure']
#print(model.docvecs.most_similar([vec]))
print(model.docvecs.most_similar('_'.join(doctag),topn=20))
#model.save('C:\\Users\\91965\\Desktop\\iMDb Database\\modell')
#neww=model.infer_vector(preprocess('Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.'))
#print(model.docvecs.most_similar(positive=[neww],topn=5))
#print(model.infer_vector(preprocess(documents[0])))
# check for wmd map creation 
#x=model.docvecs.doctag_syn0
#wmd=np.zeros(x.shape)
#for i in range(x.shape[0]):
#	for j in range(x.shape[1]):
#		wmd[i][j]=model.wmdistance(preprocess(documents[i]),preprocess(documents[j]))
#print('d')		
distortions = []
K = range(2,15)
#for k in K:
#    kmeanModel = KMeans(n_clusters=k)
#    kmeanModel.fit(x)
#    distortions.append(kmeanModel.inertia_)
   	
#    	labels = kmeanModel.labels_
#    	print(labels)
		#if(k>1):
			#kclusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance)
			#assigned_clusters = kclusterer.cluster(x, assign_clusters=True)

#for k in range(2,15):
#	clustering = AgglomerativeClustering(n_clusters=k,linkage='average',affinity='cosine')
#	clustering.fit(x)
#	labels=clustering.labels_
	
#	distortions.append(metrics.silhouette_score(x, labels,metric='cosine'))
#s="When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."
#print(model.docvecs.most_similar(8974,topn=5))
#print(model.docvecs.most_similar(7356,topn=5))
##print(documents[388])
print('\n')
import matplotlib.pyplot as plt
import seaborn as sns
from tsne import bh_sne

def pca_transform_vecs(vecs, n=50):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca.fit(vecs)
    return pca.transform(vecs)
from numpy import float64
def tsne_plot(titless, dims=40, perplexity=6, save_as=''):
    doctags = ['_'.join(title(tittle)) for tittle in titless]
    vecs = np.array([model.docvecs[doctag] for doctag in doctags])    
    # First doing pca on the vectors can reduce the noise and yield a better
    # 2d projection
    small_vecs = pca_transform_vecs(vecs, dims)
    tsne_vecs = bh_sne(float64(small_vecs), perplexity=perplexity)

    fig, ax = plt.subplots(figsize=(24, 24))
    ax.scatter(tsne_vecs[:,0], tsne_vecs[:,1])

    # Annotate points with the movie title
    for i, tittle in enumerate(titless):
        ax.annotate(tittle, 
                    xy=(tsne_vecs[i,0],tsne_vecs[i,1]), 
                    fontsize=12, alpha=.9)

    plt.xlim(min(tsne_vecs[:,0])-0.3, max(tsne_vecs[:,0])+0.3) 
    plt.ylim(min(tsne_vecs[:,1])-0.3, max(tsne_vecs[:,1])+0.3)
    if save_as:
        fig.savefig(save_as, dpi=fig.dpi)
    return fig
fig = tsne_plot(titles[-300:], save_as='movies.png')    
##print(documents[10431])
#print('\n')
#print(documents[4234])
#print('\n')
#print(documents[11026])
#print('\n')
#plt.figure(figsize=(16,8))
#plt.plot(K, distortions, 'bx-')
#plt.xlabel('k')
#plt.ylabel('Distortion')
#plt.title('The Elbow Method showing the optimal k')
#plt.show()

