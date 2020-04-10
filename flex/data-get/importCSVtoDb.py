import pandas as pd
import mysql.connector

file_ = "data.tsv"
data = pd.read_csv(file_, delimiter='\t', encoding='utf-8')
region = data.to_numpy()

conn = mysql.connector.connect(host='localhost',
                               database='flex',
                               user='root',
                               password='')


mycursor = conn.cursor()
initQuery="select * from mytable"
mycursor.execute(initQuery)
moviesDB = mycursor.fetchall()
length = len(moviesDB)
locales = ['US','UK', 'IN', 'AU']
j=0
i=0
for i in range(len(region)):
	flag=0
	if(region[i][1]==moviesDB[j][0]):
		print("Match")
		k=i+1
		while(region[k][1]==region[i][1] and flag==0):
			if(moviesDB[j][3]==region[k][2] and region[k][3] in locales):
				flag=1
			else:
				k=k+1
		i=k
		
		if(flag==1):
			cursor.execute('INSERT INTO `cleantable`(`tconst`, `titleType`, `primaryTitle`, `originalTitle`, `isAdult`, `startYear`, `runtimeMinutes`, `genres`, `averageRating`, `numVotes`)' 
				'VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)', 
				(moviesDB[j][0],moviesDB[j][1],moviesDB[j][2],moviesDB[j][3],moviesDB[j][4],moviesDB[j][5],moviesDB[j][6],moviesDB[j][7],moviesDB[j][8],moviesDB[j][9]))
			print("Value Inserted")
		j=j+1
	else:
		i=i+1
	print(i)


mycursor.close()
conn.close()