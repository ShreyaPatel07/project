import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import sqlite3
import pandas as pd
import cv2
import numpy as np
from deepface import DeepFace
from datetime import datetime
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
    except Exception as e:
        print(f"Database Error: {e}")

# Fetch Data for Display
def fetch_data():
    conn = sqlite3.connect("face_recognition.db")
    cursor = conn.cursor()
    cursor.execute("SELECT name, age, gender, emotion, timestamp FROM face_data")
    rows = cursor.fetchall()
    conn.close()
    return rows

# Export Data to CSV
def export_data():
    data = fetch_data()
    if not data:
        messagebox.showinfo("No Data", "No data available to export.")
        return
    df = pd.DataFrame(data, columns=["Name", "Age", "Gender", "Emotion", "Timestamp"])
    df.to_csv("face_data_export.csv", index=False)
    messagebox.showinfo("Success", "Data exported successfully as 'face_data_export.csv'.")

# Filter Data
def search_data():
    query = search_entry.get().strip().lower()
    for item in tree.get_children():
        tree.delete(item)
    for row in fetch_data():
        if query in row[0].lower() or query in row[3].lower():
            tree.insert("", "end", values=row)

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
            if w < 60 or h < 60:
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
    # Ask for the user's name
    name = simpledialog.askstring("Input", "Enter Name (or leave blank for Unknown):")
    if not name:
        name = "Unknown"

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            if w < 60 or h < 60:
                continue
            
            face_img = frame[y:y+h, x:x+w]

            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
                analyse = analyse[0] if isinstance(analyse, list) else analyse

                age = int(analyse["age"])
                gender = str(analyse["gender"])
                
                # gender_raw = str(analyse["gender"]).lower()
                # gender = "Woman" if "female" in gender_raw else "Man" 

                # gender_scores = analyse["gender"]
                # gender_label = max(gender_scores, key=gender_scores.get)

                # # Normalize
                # if gender_label.lower() == "woman":
                #     gender = "Woman"
                # elif gender_label.lower() == "man":
                #     gender = "Man"
                # else:
                #     gender = "Unknown"
               


                # print("Detected gender from DeepFace:", analyse["gender"])


               
               
                emotion = str(max(analyse["emotion"], key=analyse["emotion"].get))

                display_text = f"Name: {name}, Age: {age}, Gender: {gender}, Emotion: {emotion}"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                # Save detected face data with entered name
                insert_data(name, age, gender, emotion)

            except Exception as e:
                print(f"Error: {e}")
                continue

        cv2.imshow("Face Recognition", frame)

        # Press 'q' to quit properly
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Tkinter UI Layout
root = tk.Tk()
root.title("Face Detection System")
root.geometry("800x800")

title_label = tk.Label(root, text="Face Detection System", font=("Helvetica", 20, "bold"))
title_label.pack(pady=10)

# Search Bar
search_frame = tk.Frame(root)
search_frame.pack(pady=5)
search_entry = tk.Entry(search_frame, width=30)
search_entry.grid(row=0, column=0, padx=5)
search_btn = tk.Button(search_frame, text="Search", command=search_data)
search_btn.grid(row=0, column=1, padx=5)

# Data Table
tree = ttk.Treeview(root, columns=("Name", "Age", "Gender", "Emotion", "Timestamp"), show="headings")
for col in ["Name", "Age", "Gender", "Emotion", "Timestamp"]:
    tree.heading(col, text=col)
    tree.column(col, width=100)

tree.pack(pady=10, fill="both", expand=True)

# Data Display
for row in fetch_data():
    tree.insert("", "end", values=row)

# Name Entry Field
name_frame = tk.Frame(root)
name_frame.pack(pady=5)
tk.Label(name_frame, text="Name:").grid(row=0, column=0, padx=5)
name_entry = tk.Entry(name_frame, width=30)
name_entry.grid(row=0, column=1, padx=5)


btn_create_dataset = tk.Button(root, text="Create Dataset", command=create_dataset, width=20, bg="blue", fg="white")
btn_create_dataset.pack(pady=5)

btn_recognize_faces = tk.Button(root, text="Recognize Faces", command=recognize_faces, width=20, bg="red", fg="white")
btn_recognize_faces.pack(pady=5)

export_btn = tk.Button(root, text="Export to CSV", command=export_data, bg="green", fg="white")
export_btn.pack(pady=5)

btn_quit = tk.Button(root, text="Quit", command=root.quit, width=20, bg="black", fg="red")
btn_quit.pack(pady=5)

initialize_db()
root.mainloop()
