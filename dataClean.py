import gzip
import pandas as pd 
import numpy as np

# names = gzip.open('title.basics.tsv.gz' ,'rb')
names=ratings='C:\\Users\\91965\\Desktop\\iMDb Database\\data.tsv'
nameBasics = pd.read_csv(names,delimiter='\t',encoding='utf-8')
# print(nameBasics)
# ratings = gzip.open('title.ratings.tsv.gz','rb')
ratings=ratings='C:\\Users\\91965\\Desktop\\iMDb Database\\ratingClean.tsv'
titleRatings = pd.read_csv(ratings,delimiter='\t',encoding='utf-8')

s = pd.DataFrame()
#s.columns=[nameBasics.columns,'averageRating','numVotes']

for index,row in titleRatings.iterrows():
	tconst = row['tconst']
	temp = nameBasics.loc[nameBasics['tconst'] == tconst]
	
	
	result = temp.empty
	
	if(result=='True'):
		continue
	tempp=((temp['titleType']=='movie') & (temp['startYear']!='\\N') & (temp['startYear'].astype(int)>=1960)).bool()
	
	if(tempp):
			
		rating=row['averageRating']
		votes=row['numVotes']
		temp['averageRating']=rating
		temp['numVotes']=votes
		
	#print(temp)
		s=s.append(temp)
		
	#print(s,"\n\n\n")
s.to_csv('C:\\Users\\91965\\Desktop\\iMDb Database\\cleaned.csv')
