import pandas as pd 
import numpy as np

ratings='data-ratings-raw.tsv'
titleRatings = pd.read_csv(ratings,delimiter='\t',encoding='utf-8')
resultSet = pd.DataFrame(columns=titleRatings.columns)
count=0
i=0

for index,row in titleRatings.iterrows():
	count=count+1
	if(row['numVotes']>=2000 and row['averageRating']>=5):
		i=i+1
		resultSet = resultSet.append({'tconst' : row['tconst'], 'numVotes' : row['numVotes'], 'averageRating' : row['averageRating']}, ignore_index=True)

	print("Count : ",count , " && Index : ",i)

print(resultSet)
with open('cleanRatings.csv', 'w') as outfile:
	outfile.write(resultSet.to_csv(index=False))

