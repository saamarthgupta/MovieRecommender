import mysql.connector
import json

conn = mysql.connector.connect(host='localhost',
                               database='flex',
                               user='root',
                               password='')
if conn.is_connected():
    print('Connected to MySQL database')
else:
    print("Error")
    die()

mycursor = conn.cursor()
i=1
with open('../db/finalMasterDatawithProdInfo.json') as json_file: 
    data=json.load(json_file)
    for line in data:
    	# plot = line['plot']
    	ids = str(line['id'])
    	title = line['title']
    	votes = line['votes']
    	avgRating = line['avgRating']
    	year = line['year']
    	genres = ','.join(line['genres'])
    	productionInfo = ','.join(line['productionInfo'])
    	mycursor.execute('INSERT INTO `data_country_info`(`id`,`title`,`votes`,`avgRating`,`year`,`genres`,`productionInfo`)'
    					'VALUES(%s, %s, %s, %s, %s, %s, %s)',
    						(ids,title,votes,avgRating,year,genres,productionInfo))
    	print(i," - inserted")
    	i=i+1