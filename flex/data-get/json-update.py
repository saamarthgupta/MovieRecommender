import pandas as pd
import simplejson as json
import mysql.connector


class DecimalEncoder(json.JSONEncoder):
    def _iterencode(self, o, markers=None):
        if isinstance(o, decimal.Decimal):
            # wanted a simple yield str(o) in the next line,
            # but that would mean a yield on the line with super(...),
            # which wouldn't work (see my comment below), so...
            return (str(o) for o in [o])
        return super(DecimalEncoder, self)._iterencode(o, markers)


conn = mysql.connector.connect(host='localhost',
                               database='flex',
                               user='root',
                               password='')
if conn.is_connected():
    print('Connected to MySQL database')
else:
    print("Error")

mycursor = conn.cursor()

masterResult = []

count = 0
temp = 0
with open('resultIdPlots.json') as data_file:
    data = json.load(data_file)
    for v in data:
        temp = temp+1
        tconst = 'tt'+v['id']
        if(v['plot'] != "null"):
            query = "SELECT * FROM moviedb where tconst='"+tconst+"'"
            mycursor.execute(query)
            myresult = mycursor.fetchone()
            count = count+1
            genres = myresult[7].split(",")
            year = myresult[5]
            tempMaster = {'id': v['id'], 'plot': v['plot'], 'title': myresult[3],
                          'genres': genres, 'year': year, 'avgRating': myresult[8], 'votes': myresult[9]}
            masterResult.append(tempMaster)

        else:

            query = "DELETE FROM moviedb where tconst='"+tconst+"'"
            mycursor.execute(query)
            conn.commit()
            print(mycursor.rowcount, "record(s) deleted")

print("Temp=", temp)
print("\n\n\n", count)
with open('finalMasterData.json', 'w') as outfile:
    json.dump(masterResult, outfile, cls=DecimalEncoder)
