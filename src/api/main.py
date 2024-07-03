
from fastapi import FastAPI, UploadFile
from tensorflow.keras.models import load_model
from tensorflow.keras import models
from fastapi import FastAPI, Request, Form
from starlette.responses import FileResponse
from fastapi import FastAPI
from fastapi import UploadFile, File
import tensorflow as tf
from fastapi.responses import JSONResponse
import cv2
from io import BytesIO
import os
import uuid
import tempfile
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware 
import shutil
from keras.applications.densenet import preprocess_input
# from moviepy.editor import VideoFileClip


import numpy as np
from PIL import Image
import io

import tensorflow as tf
from fastapi import FastAPI, UploadFile, File
import os


app = FastAPI()
# Load the pre-trained deepfake model
model = None

# Load the pre-trained deepfake model
model_path = r'C:\Users\Hafiz Pc\Desktop\myfirstapp\fastapi\myenv\results'

try:
    # Attempt to load the model
    model = tf.saved_model.load(model_path)
except ValueError as e:
    # Handle the ValueError exception (file format not supported)
    print(f"Error loading model: {e}")

@app.post('/prediction')
async def prediction(file: UploadFile = File(...)):
    if model is None:
        # Model is not loaded, return an error response
        return {"error": "Model is not loaded"}

    # Your prediction logic here
    pass

# Create the FastAPI app
app = FastAPI()
origins = [
   
    "http://localhost:3000",
    
    
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/savevideo')
async def root(file: UploadFile = File(...)):
    with open ('test.mp4', "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

@app.get('/test')
def read_root(request: Request):
    return {"Welcome"}

@app.get('/')
def form_post(request: Request):
    return {"Welcome"}

def save_uploaded_file(upload_file, destination_folder):
    # Create a temporary file to save the uploaded video
    temp_file_path = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
    with open(temp_file_path, "wb") as temp_file:
        shutil.copyfileobj(upload_file.file, temp_file)

    # Move the temporary file to the destination folder
    video_filename = upload_file.filename
    video_file_path = os.path.join(destination_folder, video_filename)
    shutil.move(temp_file_path, video_file_path)

    return video_file_path

@app.post('/prediction')
async def prediction(file: UploadFile = File(...)):
    content_type = file.content_type

    if 'image' in content_type:
        # Image prediction
        image = Image.open(io.BytesIO(await file.read()))
        # Convert the image to a format suitable for OpenCV
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Load the pre-trained Haar Cascade face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(cv_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return JSONResponse(content={"predicted_class": "No face detected"})

        # Assuming one face per image, take the first detected face
        x, y, w, h = faces[0]
        face_image = image.crop((x, y, x+w, y+h))
         # Save the extracted face frame in the destination folder
        destination_folder = destination_folder = r"C:\Users\Hafiz Pc\Desktop\myfirstapp\fastapi\myenv\myenv\destination"
  # Set the desired destination folder path
        os.makedirs(destination_folder, exist_ok=True)
        face_filename = f"face_{uuid.uuid4().hex}.jpg"
        face_filepath = os.path.join(destination_folder, face_filename)
        face_image.save(face_filepath)
        pred = preprocess_image(image)
        predResult = 'Fake' if pred == 0 else 'Real'
        print(predResult)

        return JSONResponse(content={"predicted_class": predResult})
    
    elif 'video' in content_type:
        # Video prediction
        destination_folder = r"C:\Users\Hafiz Pc\Desktop\myfirstapp\fastapi\myenv\myenv\destination"  

        # Save the uploaded video to a temporary file first
        temp_video_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.mp4")
        with open(temp_video_path, "wb") as temp_file:
            shutil.copyfileobj(file.file, temp_file)

        # Extract frames and save them in the destination folder
        frames = extract_frames(temp_video_path, destination_folder)
        predictions = []

        for frame in frames:
            pred = preprocess_image(frame)
            predictions.append(pred)

        predicted_class = max(set(predictions), key=predictions.count)
        predResult = 'Video_fake' if predicted_class == 0 else 'Video_real'

        return {"predicted_class": predResult}
    
    else:
        return {"error": "Invalid file type"}
    

def preprocess_image(image):
    resized_image = image.resize((128, 128))
    x = tf.keras.utils.img_to_array(resized_image)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x).argmax()
    return predictions



def extract_frames(file, destination_folder):
    os.makedirs(destination_folder, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    frames = []
    video = cv2.VideoCapture(file)

    frame_count = 0
    while True:
        success, frame = video.read()

        if not success:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Crop and resize the frame containing the detected face to 128x128 pixels
            x, y, w, h = faces[0]
            face_frame = frame[y:y+h, x:x+w]
            pil_image = Image.fromarray(cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB))
            resized_frame = pil_image.resize((128, 128))

            # Save the frame as an image in the destination folder
            frame_filename = f"{frame_count:04d}.jpg"  # Use frame count as the filename
            frame_filepath = os.path.join(destination_folder, frame_filename)
            resized_frame.save(frame_filepath)

            frames.append(resized_frame)

        frame_count += 1

    video.release()

    return frames






  



 