import cv2 as cv
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import pickle
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder
from mtcnn.mtcnn import MTCNN

# ===================== LOAD MODELS =====================
facenet = FaceNet()
detector = MTCNN()

# Load embeddings labels
faces_embedding = np.load(
    r"C:\Users\Shishupal Kumar\OneDrive\Desktop\python-project\12_face_prediction_in_attendance\UI\faces_embeddings_done_4classes.npz"
)

Y = faces_embedding['arr_1']

encoder = LabelEncoder()
encoder.fit(Y)

# Load SVM model
model = pickle.load(
    open(
        r"C:\Users\Shishupal Kumar\OneDrive\Desktop\python-project\12_face_prediction_in_attendance\UI\svm_model_160x160.pkl",
        "rb"
    )
)

# ===================== CAMERA =====================
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Detect faces using MTCNN
    faces = detector.detect_faces(rgb_img)

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)

        face_img = rgb_img[y:y+h, x:x+w]

        if face_img.size == 0:
            continue

        face_img = cv.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)

        # FaceNet embedding
        embedding = facenet.embeddings(face_img)

        # Predict using SVM
        ypred = model.predict(embedding)
        final_name = encoder.inverse_transform(ypred)[0]

        # Draw results
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.putText(
            frame,
            final_name,
            (x, y-10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv.imshow("Face Recognition (MTCNN + FaceNet)", frame)

    if cv.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
