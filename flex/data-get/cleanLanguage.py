import pandas as pd
import mysql.connector

file_ = "data.tsv"
data = pd.read_csv(file_, delimiter='\t', encoding='utf-8')
conn = mysql.connector.connect(host='localhost',
                               database='flex',
                               user='root',
                               password='')
if conn.is_connected():
    print('Connected to MySQL database')
else:
    print("Error")

mycursor = conn.cursor()

query = "delete from mytable where id="
i=0

for index, row in data.iterrows():
    isOriginal = row['isOriginalTitle']
    if(isOriginal == 1):
        locale = row['region']
        if(locale == 'US' or locale == 'IN' or locale == "UK"or locale == "AU" or locale == "\\N"):
            print(i)
            i=i+1
        else:
            try:
                deleteQuery = query+"'"+row['titleId']+"'"
                mycursor.execute(deleteQuery)
                conn.commit()
                print(mycursor.rowcount, "record(s) deleted - ", row['title'])
            except:
            	found=0

mycursor.execute(query)
movies = mycursor.fetchall()
