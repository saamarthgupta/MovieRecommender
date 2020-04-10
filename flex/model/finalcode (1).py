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
from sklearn.cluster import DBSCAN 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import sys
from multiprocessing import cpu_count
from gensim.models.doc2vec import LabeledSentence
import string
import re

RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def preprocess(text):
    
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Return the cleaned text as a list of words
    4. Remove words
    
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
   
    return [stemmer.lemmatize(word) for word in text]
    
def title(text):

    # Remove all punctuation and make all lowercase 
    return RE_PUNCT.sub(" ", text).lower().split()
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)  


titles={}
with open('finalMasterDatawithProdInfo.json') as json_file: 
	data=json.load(json_file)
	for line in data:
		titles[line['title']]= '_'.join(title(line['title'])+[str(line['year'])])
	documents=pd.read_json('finalMasterDatawithProdInfo.json')
	documents.set_index('title',inplace=True)

fname ='C:\\Users\\91965\\Desktop\\iMDb Database\\trippin3'
model = Doc2Vec.load(fname)    

def recommend(p):
    
    
    ids=[None]*20
    name=[None]*20
    score=[None]*20
    final=[]
    ans=model.docvecs.most_similar([titles[p]],topn=20)
    for i in range(20):
        name[i]= list(titles.keys())[list(titles.values()).index(ans[i][0])]
        ids[i]=documents.loc[name[i]]['id']
        score[i]=ans[i][1]
    final=list(zip(name,list(zip(ids,score))))
    return final

movie=input("Search for movie\n")  

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 1),min_df=0, stop_words='english')
print(" ".join(preprocess(documents.iloc[0]['genres'])))
tfidf_matrix = tf.fit_transform([" ".join(preprocess(documents.iloc[i]['genres'])) for i in range(documents.shape[0])])
print(tf.get_feature_names())

indices={}
for i in range(documents.shape[0]):
    indices['_'.join(title(documents.index[i]))]=i   
def genre_similarity(movie1,movie2):
    return metrics.pairwise.cosine_similarity(tfidf_matrix[indices['_'.join(title(movie1))]],tfidf_matrix[indices['_'.join(title(movie2))]])

#print(genre_similarity('Gully boy',''))
recommendations=recommend(movie)
