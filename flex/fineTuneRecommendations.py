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

# Load Model
# fname ='model/model'
fname = 'model/model2'
model = Doc2Vec.load(fname)
masterFile = 'db/finalMasterDatawithProdInfo.json'
with open('model/tfidf_matrix', 'rb') as tfidf_matrixFile:
		tfidf_matrix = pickle.load(tfidf_matrixFile) 
with open('model/tfidf_indices', 'rb') as tfidf_indices:
		indices = pickle.load(tfidf_indices)

with open(masterFile) as json_file: 
    data=json.load(json_file)
    for line in data:
    	titles[line['title']]= '_'.join(title(line['title'])+[str(line['year'])])
    documents=pd.read_json(masterFile)
    documents.set_index('title',inplace=True)

searchString = input("Search for movie : ")
matchedMovies = searchMovies(searchString, conn, mycursor)
if(len(matchedMovies)==0):
	print("No Movie Found in DB.")
	exit()

movieInfo = getMovieInfo(matchedMovies)
print("Movie Info - ", movieInfo)
movieTitle = str(movieInfo[1])
# print("Movie Title = ", movieTitle, " (", movieInfo[4], ")")

modelRecommendationResults = list(modelRecommendations.recommend(movieTitle, titles, model, documents))
modelRecommendationResults = np.array(modelRecommendationResults)
# print(modelRecommendationResults)

defaultCountrySet = ['United States','India','Australia','United Kingdom']

# Normalise Similarity scores By Dividing with max of Similarity Score
normaliseSimScores(modelRecommendationResults)
maxGenSim = getMaxGenSim(modelRecommendationResults, movieInfo[1])
for movie in modelRecommendationResults:
	movieId = movie[1]
	# print(movieId)

	found=True
	movieData=[]
	try:
		query = "SELECT * from `data_country_info` where id=" + movieId
		mycursor.execute(query)
		movieData = mycursor.fetchone()
	except:
		print("None Found")
		found=False
	
	if movieData is None or found is False:
		continue		

	countrySim = getCountrySimilarity(movieData[5].split(','), defaultCountrySet)
	yearSim = getYearSimilarity(movieData,movieInfo)
	# print("MG - ", movieData[0],"\t","MD - ", movieInfo[0], getGenreSimilarity(movieData[1],movieInfo[1]))
	# Normalise Genre Similarity Score
	genreSim = getGenreSimilarity(movieData[1],movieInfo[1])/maxGenSim

	# try:
	# 	query = "SELECT `productionInfo` from `data_country_info` where id=" + movieId
	# 	mycursor.execute(query)
	# 	countryString=mycursor.fetchone()
	# 	# print("Len : ", len(countryString))
	# 	if(len(countryString)>0):
	# 		movieCountrySet = countryString[0].split(',')
	# except:
	# 	# print("Empty Result")	
	# 	pass
	# if(len(movieCountrySet)==0):
	# 	countrySim=0
	# else:
	# 	# print("movieCountrySet : ", movieCountrySet)
	# 	countrySim = getCountrySimilarity(movieCountrySet,defaultCountrySet)

	movie[2] = 0.55 * float(movie[2]) + 0.45 * countrySim + 0.15 * yearSim + 0.4 * float(genreSim[0])

# modelRecommendationResults.sort(key = lambda x:x[2]) # 0 - Title, 1 -Index, 2 - Sim Score
modelRecommendationResults = modelRecommendationResults[modelRecommendationResults[:,2].argsort()]
#Reverse Sorted Array
modelRecommendationResults = modelRecommendationResults[::-1] 

print(modelRecommendationResults)




