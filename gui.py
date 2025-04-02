import os
import cv2  # type: ignore # OpenCV for image processing
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.models import model_from_json # type: ignore
from PIL import Image, ImageTk # type: ignore
import tkinter as tk
from tkinter import filedialog, Label, Button

# Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
tf.get_logger().setLevel("ERROR")

# Load Haar cascade for face detection
haar_cascade_path = "haarcascade_frontalface_default.xml"
if not os.path.exists(haar_cascade_path):
    print("⚠️ Haar cascade file not found! Download it from OpenCV.")
face_cascade = cv2.CascadeClassifier(haar_cascade_path)

# Emotion categories
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

# Function to load the model
def load_model(json_file, weights_file):
    try:
        with open(json_file, "r") as file:
            loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(weights_file)  
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

# Load the trained model
model = load_model("model_a.json", "model_weights1.weights.h5")

if model is None:
    print("❌ Model loading failed. Exiting...")
    exit()

# Initialize GUI
top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial', 15, 'bold'))
sign_image = Label(top)

def detect_emotion(file_path):
    """Detects face and predicts emotion"""
    image = cv2.imread(file_path)
    if image is None:
        label1.configure(foreground="#011638", text="❌ Unable to load image")
        return

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    cv2.imshow("Grayscale Image", gray_image)  # Debugging: Show the grayscale image
    cv2.waitKey(500)  # Pause for debugging
    cv2.destroyAllWindows()

    # Detect faces with improved parameters
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

    if len(faces) == 0:
        label1.configure(foreground="#011638", text="⚠️ No face detected")
        return

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw rectangle around face
        face_crop = gray_image[y:y + h, x:x + w]  # Crop face
        face_resized = cv2.resize(face_crop, (48, 48))  # Resize to match model input
        face_resized = np.expand_dims(face_resized, axis=0)  # Add batch dimension
        face_resized = np.expand_dims(face_resized, axis=-1)  # Add channel dimension

        # Predict emotion
        if model is not None:
            prediction = model.predict(face_resized)
            emotion_label = EMOTIONS_LIST[np.argmax(prediction)]  # Get the predicted emotion
            label1.configure(foreground="#011638", text=f"Detected Emotion: {emotion_label}")
            print(f"✅ Detected Emotion: {emotion_label}")

    # Show detected faces
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(1)  # Prevent GUI freezing

def show_detect_button(file_path):
    detect_button = Button(top, text="Detect Emotion", command=lambda: detect_emotion(file_path), padx=10, pady=5)
    detect_button.configure(background="#364156", foreground='white', font=('arial', 10, 'bold'))
    detect_button.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if not file_path:
            return

        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width() / 2.25), (top.winfo_height() / 2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        print(f"❌ Error uploading image: {e}")
        label1.configure(text="Error uploading image")

upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156", foreground='white', font=('arial', 20, 'bold'))
upload.pack(side='bottom', pady=50)
sign_image.pack(side='bottom', expand=True)
label1.pack(side='bottom', expand=True)

top.mainloop()
