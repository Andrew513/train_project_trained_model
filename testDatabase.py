import sqlite3

#connecting database
conn = sqlite3.connect('boundingBox.db')
c = conn.cursor()

#c.execute("CREATE TABLE bbox (class_id int,x1 float,y1 float,x2 float,y2 float)")
class_id = 50
x1 = 152
y1 = 242
x2 = 121
y2 = 175

c.execute("INSERT INTO bbox (class_id, x1, y1, x2, y2) VALUES(?, ?, ?, ?, ?)",(class_id, x1, y1, x2, y2))

conn.commit()
c.close()
conn.close()
