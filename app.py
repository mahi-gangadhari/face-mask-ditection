import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
from PIL import Image

# Load pre-trained model
model = load_model("mask_detector.h5")

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

IMG_SIZE = 28  # Your model expects 28x28 grayscale input

# Preprocessing function
def preprocess_face(image):
    img_np = np.array(image.convert("RGB"))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        return None, None, "❌ No face detected."

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]
    face_resized = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
    face_normalized = face_resized.astype("float32") / 255.0
    face_flattened = face_normalized.flatten().reshape(1, -1)  # Shape (1, 1600)

    # Optional: Draw a box
    cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return face_flattened, img_np, None


def predict_label(face_input):
    pred = model.predict(face_input)[0]
    return "😷 Wearing Mask" if pred[1] > pred[0] else "❌ No Mask"


# Streamlit UI
st.title("😷 Face Mask Detection App")
uploaded_file = st.file_uploader("Upload an image (jpg/jpeg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    face_input, marked_image, error = preprocess_face(image)

    if error:
        st.error(error)
    else:
        label = predict_label(face_input)
        st.success(f"Prediction: **{label}**")
        st.image(marked_image, caption="Detected Face", use_column_width=True)




