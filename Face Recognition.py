import cv2
import numpy as np
import sqlite3
import os
from PIL import Image
import time

DB_PATH = "D:/Coding/Python/Face Recognition/database.db"
DATASET_PATH = "D:/Coding/Python/Face Recognition/dataset"
RECOGNIZER_PATH = "D:/Coding/Python/Face Recognition/recognizer/trainingdata.yml"
CASCADE_PATH = "D:/Stuff/Python 2/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml"

def create_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS STUDENTS (Id INTEGER PRIMARY KEY, Name TEXT, Age INTEGER)")
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def insert_or_update(Id, Name, age):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM STUDENTS WHERE Id=?", (Id,))
    isRecordExists = cursor.fetchone()
    if isRecordExists:
        cursor.execute("UPDATE STUDENTS SET Name=?, Age=? WHERE Id=?", (Name, age, Id))
        print("Record updated successfully.")
    else:
        cursor.execute("INSERT INTO STUDENTS (Id, Name, Age) VALUES (?, ?, ?)", (Id, Name, age))
        print("New record created successfully.")
    conn.commit()
    conn.close()

def get_profile_by_id(Id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM STUDENTS WHERE Id=?", (Id,))
    profile = None
    for row in cursor:
        profile = row
    conn.close()
    return profile

def delete_entry(Id):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM STUDENTS WHERE Id=?", (Id,))
    conn.commit()
    conn.close()
    print(f"Deleted user with ID: {Id}")
    
    for file in os.listdir(DATASET_PATH):
        if file.startswith(f"user.{Id}."):
            os.remove(os.path.join(DATASET_PATH, file))
            print(f"Deleted file: {file}")

def capture_faces_from_webcam(Id):
    faceDetect = cv2.CascadeClassifier(CASCADE_PATH)
    video_capture = cv2.VideoCapture(0)
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    sampleNum = 0
    print("Capturing face samples. Press ESC to exit early.")
    
    while True:
        ret, img = video_capture.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            sampleNum += 1
            cv2.imwrite(f"{DATASET_PATH}/user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.waitKey(100)

        cv2.imshow("Capturing Faces", img)

        key = cv2.waitKey(1)
        if key == 27 or sampleNum >= 20:
            break

    video_capture.release()
    cv2.destroyAllWindows()
    print(f"Captured {sampleNum} face samples.")

def process_existing_image(image_path, user_id, sample_count=20):
    faceDetect = cv2.CascadeClassifier(CASCADE_PATH)
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return False
    
    for i, (x, y, w, h) in enumerate(faces):
        face_img = gray[y:y+h, x:x+w]
        
        for sample_num in range(1, sample_count+1):
            if sample_num > 1:
                noise = np.random.normal(0, 5, face_img.shape).astype(np.uint8)
                face_variation = cv2.add(face_img, noise)
                
                if sample_num % 3 == 0:
                    rows, cols = face_variation.shape
                    angle = np.random.uniform(-5, 5)
                    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
                    face_variation = cv2.warpAffine(face_variation, M, (cols, rows))
            else:
                face_variation = face_img
            
            output_path = f"{DATASET_PATH}/user.{user_id}.{sample_num}.jpg"
            cv2.imwrite(output_path, face_variation)
    
    print(f"Created {sample_count} samples from image.")
    return True

def train_recognizer():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    os.makedirs(DATASET_PATH, exist_ok=True)
    os.makedirs(os.path.dirname(RECOGNIZER_PATH), exist_ok=True)
    
    if not os.listdir(DATASET_PATH):
        print("No training data found in dataset folder!")
        return False
    
    print("Training started. Press ESC to cancel.")
    
    images_path = [os.path.join(DATASET_PATH,f) for f in os.listdir(DATASET_PATH)]
    
    faces=[]
    ids=[]

    for image_path in images_path:
        faceImg=Image.open(image_path).convert('L')
        faceNp=np.array(faceImg,'uint8')
        
        id=int(os.path.split(image_path)[-1].split('.')[1])
        
        faceNp=cv2.resize(faceNp,(200,200))
        
        faces.append(faceNp)
        ids.append(id)

        cv2.imshow("Training",faceNp)

        key=cv2.waitKey(10)
        if key==27: 
            print("Training cancelled by user.")
            cv2.destroyAllWindows()
            return False

    recognizer.train(faces,np.array(ids))
    
    recognizer.save(RECOGNIZER_PATH)

    print("Training completed successfully.")
    
    cv2.destroyAllWindows()

def recognize_faces():
    if not os.path.exists(RECOGNIZER_PATH):
        print("No training data found. Please train the system first.")
        return
        
    faceDetect = cv2.CascadeClassifier(CASCADE_PATH)
    cam = cv2.VideoCapture(0)
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    try:
        recognizer.read(RECOGNIZER_PATH)
    except:
        print("Error reading training data. Please train again.")
        return
    
    print("Face recognition started. Press ESC to quit.")
    
    sky_blue = (255, 191, 0)
    
    while True:
        ret, img = cam.read()
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = faceDetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            center = (x + w//2, y + h//2)
            radius = max(w, h) // 2
            
            cv2.circle(img, center, radius, sky_blue, 2)
            
            id, _ = recognizer.predict(gray[y:y+h, x:x+w])
            profile = get_profile_by_id(id)

            if profile is not None:
                cv2.putText(img, f"Name: {profile[1]}", (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 1, sky_blue, 2)
                cv2.putText(img, f"Age: {profile[2]}", (x, y+h+45), cv2.FONT_HERSHEY_SIMPLEX, 1, sky_blue, 2)

        cv2.imshow("Face Recognition", img)

        key = cv2.waitKey(1)
        if key == 27:
            break  

    cam.release()
    cv2.destroyAllWindows()

def main_menu():
    while True:
        print("\n===== Face Recognition System =====")
        print("1. Initialize Database")
        print("2. Add User (via Webcam)")
        print("3. Add User (from Image)")
        print("4. Train Recognition Model")
        print("5. Start Face Recognition")
        print("6. Delete User")
        print("7. Run Complete Workflow (Add > Train > Recognize)")
        print("0. Exit")

        choice = input("\nEnter your choice (0-7): ")

        if choice == '1':
            create_database()

        elif choice == '2':
            Id = input("Enter user ID: ")
            Name = input("Enter user Name: ")
            age = input("Enter user Age: ")
            insert_or_update(Id, Name, age)
            capture_faces_from_webcam(Id)

        elif choice == '3':
            Id = input("Enter user ID: ")
            Name = input("Enter user Name: ")
            age = input("Enter user Age: ")
            image_path = input("Enter the full path to the person's image: ")
            insert_or_update(Id, Name, age)
            process_existing_image(image_path, Id)

        elif choice == '4':
            train_recognizer()

        elif choice == '5':
            recognize_faces() 

        elif choice == '6':
            Id = input("Enter user ID to delete: ")
            delete_entry(Id)

        elif choice == '7':
            print("\n--- Starting Complete Workflow ---")

            create_database()

            add_method = input("\nAdd user via webcam or image? (w/i): ")
            Id = input("Enter user ID: ")
            Name = input("Enter user Name: ")
            age = input("Enter user Age: ")
            insert_or_update(Id, Name, age)

            if add_method.lower() == 'w':
                capture_faces_from_webcam(Id)
            else:
                image_path = input("Enter the full path to the person's image: ")
                process_existing_image(image_path, Id)

            print("\nProceeding to train the model...")
            time.sleep(1)
            train_recognizer()

            print("\nStarting face recognition...")
            time.sleep(1)
            recognize_faces() 

        elif choice == '0':
            print("Exiting program.")
            break 

        else:
            print("Invalid choice. Please try again.")
            
if __name__=="__main__":
   main_menu()
