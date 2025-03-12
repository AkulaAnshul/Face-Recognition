import cv2
import numpy as np
import sqlite3

faceDetect = cv2.CascadeClassifier('D:/Stuff/Python 2/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

def insert_or_update(Id, Name, age):
    conn = sqlite3.connect("D:/Coding/Python/Face Recognition/database.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM STUDENTS WHERE Id=?", (Id,))
    isRecordExists = cursor.fetchone()
    if isRecordExists:
        cursor.execute("UPDATE STUDENTS SET Name=?, Age=? WHERE Id=?", (Name, age, Id))
    else:
        cursor.execute("INSERT INTO STUDENTS (Id, Name, Age) VALUES (?, ?, ?)", (Id, Name, age))
    conn.commit()
    conn.close()

Id = input("Enter user ID: ")
Name = input("Enter user Name: ")
age = input("Enter user Age: ")

insert_or_update(Id, Name, age)

sampleNum = 0
while True:
    ret, img = video_capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = faceDetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        sampleNum += 1
        cv2.imwrite(f"D:/Coding/Python/Face Recognition/dataset/user.{Id}.{sampleNum}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.waitKey(100)
    cv2.imshow("Face", img)
    cv2.waitKey(1)
    if sampleNum > 20:
        break

video_capture.release()
cv2.destroyAllWindows()