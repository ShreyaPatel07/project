import tkinter as tk
from tkinter import messagebox
import cv2
import numpy as np
from deepface import DeepFace
import os

# Create GUI window
root = tk.Tk()
root.title("Face Detection System")
root.geometry("400x400")

# Directory for dataset
DATASET_DIR = 'Dataset'
os.makedirs(DATASET_DIR, exist_ok=True)

# Function to create dataset
def create_dataset():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    person_path = os.path.join(DATASET_DIR, name)
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
            count += 1
            face_img = frame[y:y+h, x:x+w]
            cv2.imwrite(os.path.join(person_path, f"{name}_{count}.jpg"), face_img)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.imshow("Dataset Collection", frame)

        # Quit on any key press
        if cv2.waitKey(1) > 0 or count >= 50:
            messagebox.showinfo("Success", f"Saved {count} images for {name}")
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to train dataset
def train_dataset():
    embedding = {}
    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if os.path.isdir(person_path):
            embedding[person] = []
            for img_name in os.listdir(person_path):
                img_path = os.path.join(person_path, img_name)
                try:
                    embed = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)[0]["embedding"]
                    embedding[person].append(embed)
                except Exception:
                    continue

    np.save("embedding.npy", embedding)
    messagebox.showinfo("Success", "Training Completed Successfully!")

# Function to recognize faces
def recognize_faces():
    if not os.path.exists("embedding.npy"):
        messagebox.showerror("Error", "No trained data found. Please train the dataset first.")
        return

    embeddings = np.load("embedding.npy", allow_pickle=True).item()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            face_img = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            try:
                analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
                analyse = analyse[0] if isinstance(analyse, list) else analyse

                age = analyse["age"]
                gender = analyse["gender"]
                emotion = max(analyse["emotion"], key=analyse["emotion"].get)

                face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

                match, max_similarity = "Unknown", -1
                for person, embeds in embeddings.items():
                    for embed in embeds:
                        similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
                        if similarity > max_similarity:
                            max_similarity = similarity
                            match = person

                label = f"{match} ({max_similarity:.2f})" if max_similarity > 0.7 else "Unknown"
                display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"

                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            except Exception:
                continue
        



        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) > 0:
            break

    cap.release()
    cv2.destroyAllWindows()

# def recognize_faces():
#     if not os.path.exists("embedding.npy"):
#         messagebox.showerror("Error", "No trained data found. Please train the dataset first.")
#         return

#     embeddings = np.load("embedding.npy", allow_pickle=True).item()
#     cap = cv2.VideoCapture(0)

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             messagebox.showerror("Error", "Failed to capture image.")
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray, 1.3, 5)

#         for (x, y, w, h) in faces:
#             face_img = frame[y:y+h, x:x+w]
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

#             try:
#                 analyse = DeepFace.analyze(face_img, actions=["age", "gender", "emotion"], enforce_detection=False)
#                 analyse = analyse[0] if isinstance(analyse, list) else analyse

#                 # Correct Emotion Handling
#                 emotion = str(analyse.get("dominant_emotion", "Unknown"))

#                 # Extract Age and Gender
#                 age = analyse["age"]
#                 gender = analyse["gender"]

#                 # Face Recognition Logic
#                 face_embedding = DeepFace.represent(face_img, model_name="Facenet", enforce_detection=False)[0]["embedding"]

#                 match, max_similarity = "Unknown", -1
#                 for person, embeds in embeddings.items():
#                     for embed in embeds:
#                         similarity = np.dot(face_embedding, embed) / (np.linalg.norm(face_embedding) * np.linalg.norm(embed))
#                         if similarity > max_similarity:
#                             max_similarity = similarity
#                             match = person

#                 label = f"{match} ({max_similarity:.2f})" if max_similarity > 0.7 else "Unknown"
#                 display_text = f"{label}, Age: {int(age)}, Gender: {gender}, Emotion: {emotion}"

#                 cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

#             except Exception as e:
#                 print(f"Error in recognition: {e}")
#                 continue

#         cv2.imshow("Face Recognition", frame)
#         if cv2.waitKey(1) > 0:
#             break

#     cap.release()
#     cv2.destroyAllWindows()


# Tkinter UI Layout
title_label = tk.Label(root, text="Face Detection System", font=("Helvetica", 20, "bold"))
title_label.pack(pady=20)

name_label = tk.Label(root, text="Enter Name for Dataset:")
name_label.pack()
name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=5)

btn_create_dataset = tk.Button(root, text="Create Dataset", command=create_dataset, width=20, bg="blue", fg="white")
btn_create_dataset.pack(pady=5)

btn_train_dataset = tk.Button(root, text="Train Dataset", command=train_dataset, width=20, bg="green", fg="white")
btn_train_dataset.pack(pady=5)

btn_recognize_faces = tk.Button(root, text="Recognize Faces", command=recognize_faces, width=20, bg="red", fg="white")
btn_recognize_faces.pack(pady=5)

btn_quit = tk.Button(root, text="Quit", command=root.quit, width=20, bg="black", fg="white")
btn_quit.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
