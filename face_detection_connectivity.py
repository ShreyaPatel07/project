import cv2
import numpy as np
from deepface import DeepFace
import sqlite3
from datetime import datetime
import tkinter as tk
from tkinter import messagebox, simpledialog
import os

# Database Initialization
def initialize_db():
    with sqlite3.connect('face_recognition.db') as conn:
        cursor = conn.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS face_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            name TEXT,
                            age INTEGER,
                            gender TEXT,
                            emotion TEXT,
                            timestamp TEXT
                        )''')
        conn.commit()

# Insert Data into Database
def insert_data(name, age, gender, emotion):
    try:
        with sqlite3.connect('face_recognition.db', timeout=10) as conn:
            cursor = conn.cursor()

            cursor.execute('''INSERT INTO face_data (name, age, gender, emotion, timestamp)
                              VALUES (?, ?, ?, ?, ?)''', 
                           (name, int(age), gender, str(emotion), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

        print(f"Data successfully stored for {name}.")
    except Exception as e:
        print(f"Database Error: {e}")

# Create Dataset Function
def create_dataset():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    dataset_dir = "Dataset"
    os.makedirs(dataset_dir, exist_ok=True)

    person_path = os.path.join(dataset_dir, name)
    os.makedirs(person_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60:  # Ignore very small faces
                continue
            
            count += 1
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_path, f"{name}_{count}.jpg"), face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Dataset Collection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
            messagebox.showinfo("Success", f"Saved {count} images for {name}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Face Recognition Function
def recognize_faces():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60:  # Ignore very small faces
                continue
            
            face_img = frame[y:y+h, x:x+w]

            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
                analyse = analyse[0] if isinstance(analyse, list) else analyse

                # Extract Data
                name = "Shivani"
                age = int(analyse["age"])
                gender = str(analyse["gender"])  
                emotion = str(max(analyse["emotion"], key=analyse["emotion"].get))

                # Display Data on Screen
                display_text = f"Name: {name}, Age: {age}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                # Insert Data into Database
                insert_data(name, age, gender, emotion)

            except Exception as e:
                print(f"Error: {e}")
                continue

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Tkinter UI Layout
root = tk.Tk()
root.title("Face Detection System")
root.geometry("400x400")

title_label = tk.Label(root, text="Face Detection System", font=("Helvetica", 20, "bold"))
title_label.pack(pady=20)

name_label = tk.Label(root, text="Enter Name for Dataset:")
name_label.pack()
name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=5)

btn_create_dataset = tk.Button(root, text="Create Dataset", command=create_dataset, width=20, bg="blue", fg="white")
btn_create_dataset.pack(pady=5)

btn_recognize_faces = tk.Button(root, text="Recognize Faces", command=recognize_faces, width=20, bg="red", fg="white")
btn_recognize_faces.pack(pady=5)

btn_quit = tk.Button(root, text="Quit", command=root.quit, width=20, bg="black", fg="white")
btn_quit.pack(pady=10)

# Initialize Database
initialize_db()

# Run Tkinter Event Loop
root.mainloop()
