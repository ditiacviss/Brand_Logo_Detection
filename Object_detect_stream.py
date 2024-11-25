import math
import os
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"best.pt")

def predict(image_array, model):
    # Perform prediction
    results = model(image_array)
    detected_objects = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            conf = round(float(box.conf[0]), 2)  # Confidence score
            label = model.names[int(box.cls[0])]  # Class label
            detected_objects.append((label, conf, (x1, y1, x2, y2)))
            cv2.rectangle(image_array, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                image_array,
                f"{label} {conf}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
    return detected_objects, image_array


def display_logos_in_row(folder_path):
    if os.path.exists(folder_path):
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('jpg', 'png', 'jpeg'))]
        if image_files:
            cols = st.columns(6)  # Adjust the number of columns as needed
            for i, filename in enumerate(image_files):
                filepath = os.path.join(folder_path, filename)
                image = cv2.imread(filepath)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                with cols[i % len(cols)]:
                    st.image(image_rgb, caption=filename, use_column_width=True)
        else:
            st.write("No logo images found in the folder.")
    else:
        st.error("The specified logo folder does not exist.")


# Call the function in your Streamlit app
logo_folder = 'logos'

def main():
    st.title("Logo Detection (Colour Images) with YOLOv11")
    st.subheader("Available Logos:")
    display_logos_in_row(logo_folder)
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        detected_objects,annotated_image = predict(img, model)

        # Display results
        st.image(annotated_image, caption="Detected Image", channels="BGR")
        st.write("Detected Objects:")
        for obj in detected_objects:
            label, conf, bbox = obj
            st.write(f"Label: {label}, Confidence: {conf}, BBox: {bbox}")


if __name__ == "__main__":
    main()
