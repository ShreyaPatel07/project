import sqlite3

# Connect to database (or create it)
conn = sqlite3.connect('face_recognition.db')
cursor = conn.cursor()

# Create table for detected faces
cursor.execute('''
    CREATE TABLE IF NOT EXISTS face_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        age INTEGER,
        gender TEXT,
        emotion TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
''')

print("Database and table created successfully.")
conn.commit()
conn.close()
