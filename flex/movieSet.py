import mysql.connector
import pickle

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

query = "SELECT * FROM data_country_info"
mycursor.execute(query)
movieData = mycursor.fetchall()

movieSet = set()


for movie in movieData:
	movieSet.add(str(int(movie[6])))

pickIn = open("rcd.txt","rb")
set0 = pickle.load(pickIn)
# print(movieSet)
# print(set0)
ini = len(set0)

set0 = set0 - movieSet
count1 = ini-len(set0)

print(len(set0))
print(count1)
count2=0
for item in set0:
	if(len(item)>9):
		print(item)
		count2=count2+1
print(count2)
# print(z)

print("Coverage = ", (count1+2*count2)*100/len(movieSet))