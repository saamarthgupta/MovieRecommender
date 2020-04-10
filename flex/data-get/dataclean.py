import gzip
import pandas as pd 
import numpy as np

# names = gzip.open('title.basics.tsv.gz' ,'rb')
names='data-titles-raw.tsv'
nameBasics = pd.read_csv(names,delimiter='\t',encoding='utf-8')
# print(nameBasics)
# ratings = gzip.open('title.ratings.tsv.gz','rb')
ratings='cleanRatings.csv'
titleRatings = pd.read_csv(ratings,delimiter=',',encoding='utf-8')
resultSet = pd.DataFrame(columns=nameBasics.columns)
count=0
movies=titleRatings.to_numpy()
noOfMovies = titleRatings.shape[0]
i=0

for index,row in titleRatings.iterrows():
	count=count+1
	tconst = row['tconst']
	nameBasicRow = nameBasics.loc[nameBasics['tconst'] == tconst]
	
	result = nameBasicRow.empty
	
	if(result=='True'):
		continue
	nameBasicRowBool=(nameBasicRow.iloc[0]['titleType']=='movie')

	if(nameBasicRowBool):
			
		rating=row['averageRating']
		votes=row['numVotes']
		titleType = nameBasicRow.iloc[0]['titleType'].encode('utf-8')
		primaryTitle = nameBasicRow.iloc[0]['primaryTitle'].encode('utf-8')
		originalTitle = nameBasicRow.iloc[0]['originalTitle'].encode('utf-8')
		isAdult = nameBasicRow.iloc[0]['isAdult']
		startYear = nameBasicRow.iloc[0]['startYear']
		endYear = nameBasicRow.iloc[0]['endYear']
		runtimeMinutes = nameBasicRow.iloc[0]['runtimeMinutes']
		genres =  nameBasicRow.iloc[0]['genres'].encode('utf-8')

		resultSet = resultSet.append({'tconst' : tconst,'titleType' :titleType, 'primaryTitle' : primaryTitle, 'originalTitle' : originalTitle,
			'isAdult' : isAdult, 'startYear' : startYear,'endYear' : endYear, 'runtimeMinutes' : runtimeMinutes, 'genres' : genres, 
			'averageRating' : rating, 'numVotes' : votes}, ignore_index=True)
		i=i+1
	print("Count : ",count , " && Index : ",i)
	
print(resultSet)
with open('cleanData.csv', 'w') as outfile:
	outfile.write(resultSet.to_csv(index=False))
	

