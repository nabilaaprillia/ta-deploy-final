from model import build_resnet50, build_efficientnet, build_resnet50_optimized
from tensorflow.keras.applications.resnet import preprocess_input  # Gunakan yang sama untuk semua
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils 

def crop_face_with_mediapipe(image):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        results = detector.process(image)
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x_min = int(bbox.xmin * w)
            y_min = int(bbox.ymin * h)
            box_w = int(bbox.width * w)
            box_h = int(bbox.height * h)

            # Pastikan bounding box tidak keluar batas
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_min + box_w)
            y_max = min(h, y_min + box_h)

            face_crop = image[y_min:y_max, x_min:x_max]
            return face_crop
    return None

# === Sidebar ===
st.sidebar.title("🔧 Pilih Model")
model_option = st.sidebar.selectbox("Model:", ["ResNet50", "EfficientNetV2B0", "ResNet50-Optimized"])
# Info tambahan model
model_accuracies = {
    "ResNet50": 0.45,
    "EfficientNetV2B0": 0.32,
    "ResNet50-Optimized": 0.64
}
st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Info Model")
st.sidebar.markdown(f"**Model aktif:** `{model_option}`")
st.sidebar.markdown(f"**Akurasi rata-rata:** {model_accuracies.get(model_option, 0):.2%}")

use_webcam = st.sidebar.checkbox("Gunakan Webcam")

# === Load Model Sekali Saja ===
if 'model' not in st.session_state or st.session_state.model_name != model_option:
    if model_option == "ResNet50":
        model = build_resnet50()
        model.load_weights("weights/resnet50.weights.h5")
    elif model_option == "EfficientNetV2B0":
        model = build_efficientnet()
        model.load_weights("weights/efficientnetv2.weights.h5")
    elif model_option == "ResNet50-Optimized":
        model = build_resnet50_optimized()
        model.load_weights("weights/resnet50_opt.weights.h5")

    st.session_state.model = model
    st.session_state.model_name = model_option

model = st.session_state.model
labels = ['jijik', 'marah', 'netral', 'sedih', 'senang', 'takut', 'terkejut']
# === Konten Utama UI Streamlit ===
st.title("😊 Klasifikasi Emosi Wajah")

uploaded_file = st.file_uploader("Upload gambar wajah (jpg/png)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        img = Image.open(uploaded_file).convert('RGB')
        img_resized = img.resize((224, 224))
        st.image(img_resized, caption='Gambar Diupload', width=250)

    img_array = np.array(img)
    face_crop = crop_face_with_mediapipe(img_array)

    if face_crop is not None:
        face_resized = cv2.resize(face_crop, (224, 224))
        img_preprocessed = preprocess_input(face_resized)
        img_input = np.expand_dims(img_preprocessed, axis=0)

        prediction = model.predict(img_input)
        pred_label = labels[np.argmax(prediction)]

        with col2:
            st.subheader("🎯 Prediksi Emosi:")
            st.success(pred_label.upper())

            st.subheader("📊 Probabilitas:")
            st.bar_chart({labels[i]: float(prediction[0][i]) for i in range(len(labels))})
    else:
        st.error("❗ Wajah tidak terdeteksi. Coba upload gambar dengan wajah yang lebih jelas.")

        prediction = model.predict(img_input)
        pred_label = labels[np.argmax(prediction)]

    with col2:
        st.subheader("🎯 Prediksi Emosi:")
        st.success(pred_label.upper())

        st.subheader("📊 Probabilitas:")
        st.bar_chart({labels[i]: float(prediction[0][i]) for i in range(len(labels))})
else:
        st.write("📤 Silakan upload gambar terlebih dahulu untuk melihat prediksi.")

# === Webcam Mode ===
if use_webcam:
    st.subheader("📷 Webcam Klasifikasi Emosi")
    run = st.checkbox("Nyalakan Kamera")
    FRAME_WINDOW = st.image([])

    cap = cv2.VideoCapture(0)
    if run:
        ret, frame = cap.read()
        if ret:
            # Preprocess frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_crop = crop_face_with_mediapipe(img)

            if face_crop is not None:
                face_resized = cv2.resize(face_crop, (224, 224))
                img_input = np.expand_dims(preprocess_input(face_resized), axis=0)

                start = time.time()
                pred = model.predict(img_input)
                duration = time.time() - start

                label = labels[np.argmax(pred)]

                cv2.putText(img, label.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                FRAME_WINDOW.image(img)
            else:
                st.warning("❗ Wajah tidak terdeteksi oleh MediaPipe.")


                # Predict
                pred = model.predict(img_input)
                

                start = time.time()
                pred = model.predict(img_input)
                duration = time.time() - start

                label = labels[np.argmax(pred)]

                # Tambahkan teks ke gambar
                cv2.putText(img, label.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                FRAME_WINDOW.image(img)
                cap.release()