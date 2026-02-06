# import streamlit as st
# import cv2
# import numpy as np
# import pickle
# import os
# from datetime import datetime
# from mtcnn.mtcnn import MTCNN
# from keras_facenet import FaceNet

# # ================= SETTINGS =================
# CONFIDENCE_THRESHOLD = 0.75
# ATTENDANCE_FILE = "attendance.csv"

# st.set_page_config(page_title="Face Recognition System")
# st.title("📸 Face Recognition Attendance System")

# # ================= LOAD MODELS =================
# @st.cache_resource
# def load_all():
#     facenet = FaceNet()
#     detector = MTCNN()
#     svm_model = pickle.load(open("svm_model_160x160.pkl", "rb"))
#     encoder = pickle.load(open("label_encoder.pkl", "rb"))
#     return facenet, detector, svm_model, encoder

# facenet, detector, model, encoder = load_all()

# # ================= ATTENDANCE SAVE =================
# def save_attendance(name):
#     now = datetime.now()
#     date = now.strftime("%Y-%m-%d")
#     time = now.strftime("%H:%M:%S")

#     if not os.path.exists(ATTENDANCE_FILE):
#         with open(ATTENDANCE_FILE, "w") as f:
#             f.write("Name,Date,Time\n")

#     with open(ATTENDANCE_FILE, "a") as f:
#         f.write(f"{name},{date},{time}\n")

# # ================= FACE PREDICTION =================
# def recognize_face(img):
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     faces = detector.detect_faces(rgb)

#     if len(faces) == 0:
#         st.warning("❌ No face detected")
#         return img

#     for face in faces:
#         x, y, w, h = face['box']
#         x, y = abs(x), abs(y)

#         face_img = rgb[y:y+h, x:x+w]
#         face_img = cv2.resize(face_img, (160, 160))
#         face_img = np.expand_dims(face_img, axis=0)

#         embedding = facenet.embeddings(face_img)
#         probs = model.predict_proba(embedding)[0]

#         class_id = np.argmax(probs)
#         confidence = probs[class_id]

#         if confidence >= CONFIDENCE_THRESHOLD:
#             name = encoder.inverse_transform([class_id])[0]
#             save_attendance(name)
#             label = f"{name} ({confidence:.2f})"
#             color = (0, 255, 0)
#         else:
#             label = "Unknown"
#             color = (0, 0, 255)

#         cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
#         cv2.putText(img, label, (x, y-10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#     return img

# # ================= IMAGE UPLOAD =================
# st.subheader("📤 Upload Image")
# uploaded = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])

# if uploaded:
#     bytes_data = uploaded.read()
#     np_img = np.frombuffer(bytes_data, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     result = recognize_face(img)
#     st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")

# # ================= REAL-TIME CAMERA =================
# st.subheader("📷 Real-Time Camera")

# camera_photo = st.camera_input("Take a photo")

# if camera_photo:
#     bytes_data = camera_photo.getvalue()
#     np_img = np.frombuffer(bytes_data, np.uint8)
#     img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#     result = recognize_face(img)
#     st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Camera Result")

# # ================= ATTENDANCE VIEW =================
# def save_attendance(name):
#     from datetime import datetime

#     now = datetime.now()
#     date = now.strftime("%Y-%m-%d")
#     time = now.strftime("%H:%M:%S")

#     if not os.path.exists("attendance.csv"):
#         with open("attendance.csv", "w") as f:
#             f.write("Name,Date,Time\n")

#     with open("attendance.csv", "a") as f:
#         f.write(f"{name},{date},{time}\n")


#         st.success("Attendance Done")




import streamlit as st
import cv2
import numpy as np
import pickle
import os
from datetime import datetime
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet

# ================= SETTINGS =================
CONFIDENCE_THRESHOLD = 0.75
ATTENDANCE_FILE = "attendance.csv"

st.set_page_config(page_title="Face Recognition System")
st.title("📸 Face Recognition Attendance System")

# ================= SESSION STATE =================
if "attendance_done" not in st.session_state:
    st.session_state.attendance_done = False

# ================= LOAD MODELS =================
@st.cache_resource
def load_all():
    facenet = FaceNet()
    detector = MTCNN()
    svm_model = pickle.load(open("svm_model_160x160.pkl", "rb"))
    encoder = pickle.load(open("label_encoder.pkl", "rb"))
    return facenet, detector, svm_model, encoder

facenet, detector, model, encoder = load_all()

# ================= ATTENDANCE SAVE =================
def save_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w") as f:
            f.write("Name,Date,Time\n")

    with open(ATTENDANCE_FILE, "a") as f:
        f.write(f"{name},{date},{time}\n")

    st.session_state.attendance_done = True   # ✅ flag set

# ================= FACE PREDICTION =================
def recognize_face(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb)

    if len(faces) == 0:
        return img

    for face in faces:
        x, y, w, h = face['box']
        x, y = abs(x), abs(y)

        face_img = rgb[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (160, 160))
        face_img = np.expand_dims(face_img, axis=0)

        embedding = facenet.embeddings(face_img)
        probs = model.predict_proba(embedding)[0]

        class_id = np.argmax(probs)
        confidence = probs[class_id]

        if confidence >= CONFIDENCE_THRESHOLD:
            name = encoder.inverse_transform([class_id])[0]
            save_attendance(name)
            label = f"{name} ({confidence:.2f})"
            color = (0, 255, 0)
        else:
            label = "Unknown"
            color = (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    return img

# ================= IMAGE UPLOAD =================
st.subheader("📤 Upload Image")
uploaded = st.file_uploader("Upload face image", type=["jpg", "jpeg", "png"])

if uploaded:
    bytes_data = uploaded.read()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = recognize_face(img)
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Result")

# ================= REAL-TIME CAMERA =================
st.subheader("📷 Real-Time Camera")
camera_photo = st.camera_input("Take a photo")

if camera_photo:
    bytes_data = camera_photo.getvalue()
    np_img = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    result = recognize_face(img)
    st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), caption="Camera Result")

# ================= ATTENDANCE DONE HEADING =================
if st.session_state.attendance_done:
    st.markdown("## ✅ ATTENDANCE DONE")
