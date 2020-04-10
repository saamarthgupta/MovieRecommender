from imdb import IMDb
import pandas as pd
import json

# create an instance of the IMDb class
imdbInstance = IMDb()

file_ = open('finalMasterData.json')
data = json.load(file_)
noOfMovies = len(data)
print("Size of JSON : ",noOfMovies)

jsonResult = []
temp=0
start = input("Start: ") 
end = input("End: ") 
opFileName = str(input("Output File: "))
opFileName= opFileName+".json"
batch = 20
countrySet = ['United States','India','Australia','United Kingdom']

for i in range(start,end+1):
	movieId = data[i]['id']
	movieData = imdbInstance.get_movie(movieId)
	if 'countries' in movieData:
		productionInfo = movieData['countries']
		#print('Length Prod Info - ',len(productionInfo))
		# print('Length Country Set - ',len(countrySet))
		z = list((set(productionInfo))-set(countrySet))
		# print("Length of Subtraction Set - ", len(z))
		if(len(productionInfo) != len(z)):
			print(i, ' - ', str(productionInfo)[1:-1])
			data[i]['productionInfo'] = productionInfo
			jsonResult.append(data[i])
	

	if ((i%batch == 0 and i!=0) or i==noOfMovies-1):
		print("Writing ",i-batch, " to ", i)
		with open(opFileName, 'a+') as outfile:
			json.dump(jsonResult, outfile)
			jsonResult = []


