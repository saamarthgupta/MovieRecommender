from imdb import IMDb
import pandas as pd
import json
import re
# import mysql.connector

# create an instance of the IMDb class
imdbInstance = IMDb()

# conn = mysql.connector.connect(host='localhost',
#                                database='flex',
#                                user='root',
#                                password='')
# if conn.is_connected():
#     print('Connected to MySQL database')
# else:
#     print("Error")
# mycursor = conn.cursor()
# query = "SELECT * FROM mytable"
# mycursor.execute(query)
# movies = mycursor.fetchall()

file_ = open('cleanData.csv')
movies=pd.read_csv(file_,delimiter=',')
movies=movies.to_numpy()

noOfMovies = len(movies)
print(noOfMovies)
plotMasterData = []
synopsisMasterData = []

#Parameters for API
start = input("Start: ") 
end = input("End: ") 
opFileName = str(input("Output File: "))
opFileName= opFileName+".json"
batch = 50
if(end==0):
	end=noOfMovies
movieData={}
for i in range(start,end+1):
	movieId = (movies[i][0])[2:]
	movieData = imdbInstance.get_movie(movieId)

	print(i)
	if('plot' in movieData):
		plotN=0
		plotsize=0
		length = len(movieData['plot'])
		for j in range(length):
			if(len(movieData['plot'][j])>plotsize):
				plotsize=len(movieData['plot'][j])
				plotN=j
		plot=movieData['plot'][plotN]

		plot=re.sub(r"::.*$","",plot)
		
	else:
		plot="null"
	# synopsis=movieData['synopsis']
	if(plot!="null"):
		plotData = {'id' : movieId, 'plot' : plot}
		plotMasterData.append(plotData)
	# synopsisData = {'id' : movieId, 'synopsis' : synopsis}
	# synopsisMasterData.append(synopsisData)
	if ((i%batch == 0 and i!=0) or i==noOfMovies-1):
		print("Writing ",i-batch, " to ", i)
		with open(opFileName, 'a+') as outfile:
			json.dump(plotMasterData, outfile)
			plotMasterData = []


# with open('synopsis.txt', 'w') as outfile:
#     json.dump(synopsisMasterData, outfile)

# with open('plot.txt', 'w') as outfile:
#     json.dump(plotMasterData, outfile)


