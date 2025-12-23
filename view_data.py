import sqlite3

conn = sqlite3.connect('face_recognition.db')
cursor = conn.cursor()

cursor.execute("SELECT * FROM face_data")
rows = cursor.fetchall()

print("ID | Name | Age | Gender | Emotion | Timestamp")
print("-" * 60)
for row in rows:
    print(row)

conn.close()
