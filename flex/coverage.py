from imdb import IMDb
from modelRecommendations import modelRecommendations
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn.metrics as metrics
import pandas as pd
import json
import re
import string 
import numpy as np
import mysql.connector
import pickle

RE_PUNCT = re.compile('([%s])+' % re.escape(string.punctuation), re.UNICODE)

def title(text):
    # Remove all punctuation and make all lowercase 
    return RE_PUNCT.sub(" ", text).lower().split()

def getMovieInfo(matchedMovies):
	i = 1
	for movie in matchedMovies:
		print(i,". - ", matchedMovies[i-1][1],"(",matchedMovies[i-1][4],")")
		i=i+1
	movieNo = int(input("Enter Movie Number : "))
	if(movieNo <= len(matchedMovies)):
		return matchedMovies[movieNo-1]
	else:
		print("Please Enter Valid No.")

def searchMovies(searchString, conn, mycursor):
	query = "SELECT * FROM `data_country_info` WHERE title LIKE '%"+searchString+"%' ORDER BY `avgRating` DESC LIMIT 5;"
	mycursor.execute(query)
	searchResult = mycursor.fetchall()
	return searchResult

def normaliseSimScores(modelRecommendations):
	maxScore = 0.1
	for movie in modelRecommendations:
		if float(movie[2]) > maxScore:
			maxScore = float(movie[2])
	for movie in modelRecommendations:
		movie = list(movie)
		movie[2] = float(movie[2])/maxScore

def getCountrySimilarity(movieCountrySet, defaultCountrySet):

	if(len(movieCountrySet)==0):
		return 0

	z = list((set(movieCountrySet)-set(defaultCountrySet)))
	lengthofSubtractionSet = len(list((set(movieCountrySet)-set(z))))
	# print("Length of Subtraction Set = ", lengthofSubtractionSet)
	if(lengthofSubtractionSet==0):
		return 0
	elif (lengthofSubtractionSet==1 and len(movieCountrySet)==1):
		return 1
	elif (lengthofSubtractionSet==1):
		return 0.5
	elif(lengthofSubtractionSet>=2):
		return 1

def getYearSimilarity(movie,movieInfo):
	
	yearDiff = abs(movie[4]-movieInfo[4])
	if(yearDiff>=40):
		return 0.2
	else:
		return 1-(yearDiff/50)


def getGenreSimilarity(movie1, movie2):

	with open('model/tfidf_matrix', 'rb') as tfidf_matrixFile:
		tfidf_matrix = pickle.load(tfidf_matrixFile) 
	with open('model/tfidf_indices', 'rb') as tfidf_indices:
		indices = pickle.load(tfidf_indices)
	return metrics.pairwise.cosine_similarity(tfidf_matrix[indices['_'.join(title(movie1))]],tfidf_matrix[indices['_'.join(title(movie2))]])

def getMaxGenSim(modelRecommendationResults, movieTitle):
	maxRes=0.01
	for movie in modelRecommendationResults:
		temp = getGenreSimilarity(movie[0],movieTitle)
		if (temp > maxRes):
			maxRes = temp
	return maxRes

conn = mysql.connector.connect(host='localhost',
                               database='flex',
                               user='root',
                               password='')
if conn.is_connected():
    print('Connected to MySQL database')
else:
    print("Error")
    exit()

mycursor = conn.cursor()

titles = {}
defaultCountrySet = ['United States','India','Australia','United Kingdom']
# Load Model
fname ='model/model'
model = Doc2Vec.load(fname)

with open('db/finalMasterData.json') as json_file: 
    data=json.load(json_file)
    for line in data:
    	titles[line['title']]='_'.join(title(line['title']))
    documents=pd.read_json('db/finalMasterData.json')
    documents.set_index('title',inplace=True)

query = "SELECT * FROM data_country_info"
mycursor.execute(query)
movies = mycursor.fetchall()
coverageSet = set()

def getMovieRecommendations(movieTest):
	
	modelRecommendationResults = list(modelRecommendations.recommend(movieTest[1], titles, model, documents))
	modelRecommendationResults = np.array(modelRecommendationResults)
	# print(modelRecommendationResults)
	# Normalise Similarity scores By Dividing with max of Similarity Score
	normaliseSimScores(modelRecommendationResults)
	maxGenSim = getMaxGenSim(modelRecommendationResults, movieTest[1])
	for movie in modelRecommendationResults:
		movieId = movie[1]
		# print(movieId)

		found = True
		movieData = []
		try:
			query = "SELECT * from `data_country_info` where id=" + movieId
			mycursor.execute(query)
			movieData = mycursor.fetchone()
		except:
			# print("None Found")
			found = False

		if movieData is None or found is False:
			continue

		countrySim = getCountrySimilarity(movieData[5].split(','), defaultCountrySet)
		yearSim = getYearSimilarity(movieTest, movieData)
		# print("MG - ", movieData[0],"\t","MD - ", movieInfo[0], getGenreSimilarity(movieData[1],movieInfo[1]))
		# Normalise Genre Similarity Score
		genreSim = getGenreSimilarity(movieData[1], movieTest[1]) / maxGenSim
		movie[2] = 0.55 * float(movie[2]) + 0.45 * countrySim + 0.15 * yearSim + 0.4 * float(genreSim[0])
	# modelRecommendationResults.sort(key = lambda x:x[2]) # 0 - Title, 1 -Index, 2 - Sim Score
	modelRecommendationResults = modelRecommendationResults[modelRecommendationResults[:, 2].argsort()]
	# Reverse Sorted Array
	modelRecommendationResults = modelRecommendationResults[::-1]
	recommendations = modelRecommendationResults[0:5]
	return recommendations

count=1
for movie in movies:
	print(count)
	recommendations = getMovieRecommendations(movie)
	# print(recommendations)
	for rcd in recommendations:
		coverageSet.add(rcd[1])
		# print(rcd[1])
	if(count%1000 == 0):
		print("Pass Complete")
		pickleOut = open("rcd.txt","wb")
		pickle.dump(coverageSet, pickleOut)
	
	count=count+1
	print("Length of Recommended Set ", len(coverageSet))
pickleOut = open("rcd.txt","wb")
pickle.dump(coverageSet, pickleOut)

# query = "SELECT * FROM data_country_info"
# mycursor.execute(query)
# movieData = mycursor.fetchall()
# movieSet = set()

# for movie in movieData:
# 	movieSet.add(str(movie[6]))

# z = coverageSet.intersection(movieSet)
print("Coverage = ", len(coverageSet)*100/len(movieData))








