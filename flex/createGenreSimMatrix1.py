import gensim.downloader as api
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
from gensim.models.doc2vec import LabeledSentence
import string
import re
import pickle

RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)
def preprocess(text):
    
    # Takes in a string of text, then performs the following:
    # 1. Remove all punctuation
    # 2. Remove all stopwords
    # 3. Return the cleaned text as a list of words
    # 4. Remove words
    
    stemmer = WordNetLemmatizer()
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join([i for i in nopunc if not i.isdigit()])
    nopunc =  [word.lower() for word in nopunc.split() if word not in stopwords.words('english')]
   
    return [stemmer.lemmatize(word) for word in text]

def title(text):

    # Remove all punctuation and make all lowercase 
    return RE_PUNCT.sub(" ", text).lower().split()

masterFile = 'db/finalMasterDatawithProdInfo.json'
documents=pd.read_json(masterFile)
documents.set_index('title',inplace=True)
documents=documents.astype({'id': int})
documents=documents.astype({'id': str})

documents.set_index('id',inplace=True)

tf = TfidfVectorizer(analyzer='word',ngram_range=(1, 1),min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform([" ".join(preprocess(documents.iloc[i]['genres'])) for i in range(documents.shape[0])])

indices={}
for i in range(documents.shape[0]):
    indices[str(documents.index[i])]=i  

print(tfidf_matrix)
print(tf.get_feature_names())
print(indices)
with open('model/tfidf_matrix', 'wb') as tfidf_matrixFile:
  pickle.dump(tfidf_matrix, tfidf_matrixFile)

with open('model/tfidf_indices', 'wb') as tfidf_indexFile:
  pickle.dump(indices, tfidf_indexFile)