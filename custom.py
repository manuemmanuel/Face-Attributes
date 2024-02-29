import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
import tkinter as tk
from deepface import DeepFace

# Model paths
GENDER_MODEL = 'weights/deploy_gender.prototxt'
GENDER_PROTO = 'weights/gender_net.caffemodel'
FACE_PROTO = "weights/deploy.prototxt.txt"
FACE_MODEL = "weights/res10_300x300_ssd_iter_140000_fp16.caffemodel"

# Model parameters
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GENDER_LIST = ['Male', 'Female']

# Initialize face detection and gender prediction models
face_net = cv2.dnn.readNetFromCaffe(FACE_PROTO, FACE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_MODEL, GENDER_PROTO)

def detect_faces(frame, confidence_threshold=0.5):
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX, endY))
    return faces

def predict_age_gender_sentiment(frame, faces):
    for (startX, startY, endX, endY) in faces:
        face_img = frame[startY:endY, startX:endX]
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        i = gender_preds[0].argmax()
        gender = GENDER_LIST[i]
        gender_confidence = gender_preds[0][i]
        
        # Predict age and sentiment for each face
        results = DeepFace.analyze(face_img, actions=['age', 'emotion'], enforce_detection=False)
        for result in results:
            age = result['age']
            sentiment = result['dominant_emotion']
            
            # Display age, gender, and sentiment
            label_gender = f"Gender: {gender} ({gender_confidence * 100:.2f}%)"
            label_age = f"Age: {age}"
            label_sentiment = f"Sentiment: {sentiment}"
            
            cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
            
            y_offset = 30
            cv2.putText(frame, label_gender, (startX, startY - y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, label_age, (startX, startY - y_offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.putText(frame, label_sentiment, (startX, startY - y_offset + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

# Create Tkinter window
root = tk.Tk()
root.title("Face Analysis")

# Configure font
font = "Courier"

# Create a frame to hold the video feed and the graph
frame_frame = tk.Frame(root)
frame_frame.pack(padx=10, pady=10)

# Create a label to display video feed
video_label = tk.Label(frame_frame, font=(font, 10))
video_label.pack(side=tk.LEFT)

# Create a frame to hold the matplotlib graph
graph_frame = tk.Frame(frame_frame)
graph_frame.pack(side=tk.RIGHT, padx=10)

# Create matplotlib figure and axis
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas.get_tk_widget().pack()

cap = cv2.VideoCapture(0)

def update_frame():
    ret, frame = cap.read()
    if ret:
        faces = detect_faces(frame)
        predict_age_gender_sentiment(frame, faces)
        
        # Convert frame from BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert frame to grayscale
        
        # Display frame in the Tkinter window
        img = cv2.resize(gray, (640, 480))  # Resize frame
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        video_label.img = img
        video_label.configure(image=img)
        
        # Update the graph with random data
        attributes = ['Gender', 'Age', 'Sentiment']
        confidences = [np.random.random() for _ in range(len(attributes))]
        ax.clear()
        ax.plot(attributes, confidences, marker='o')  # Use plot instead of bar for line graph
        ax.set_ylim(0, 1)  # Set y-axis limit to 0-1 for confidences
        ax.set_title('Attribute Confidences', fontname=font)
        ax.set_ylabel('Confidence', fontname=font)
        ax.set_xlabel('Attribute', fontname=font)
        canvas.draw()
        
    video_label.after(10, update_frame)

# Start updating the frame
update_frame()

# Run the Tkinter event loop
root.mainloop()

# Release the camera and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()
