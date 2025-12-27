# Streamlit Face Recognition System

A web-based face recognition system built with Streamlit, TensorFlow, and OpenCV.

## Features

- Live webcam recognition
- Image upload for recognition
- Add person via webcam capture or photo upload
- Database management (view/remove registered faces)
- Adjustable recognition threshold and detection confidence

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```
## How to Use

### Live Recognition
1. Select "Live Recognition" from the sidebar
2. Capture from webcam
3. The system will detect and recognize faces

### Upload Image
1. Select "Upload Image"
2. Upload any image containing faces
3. See recognition results with confidence scores

### Add a New Person
1. Select "Add Person"
2. Enter the person's name
3. Choose either:
   - Capture from Webcam
   - Upload Photo
4. Click "Add to Database"

### Manage Database
1. Select "Database"
2. View all registered people
3. Remove people as needed

## Settings

Adjust in the sidebar:
- Recognition Threshold: Lower = more strict (default: 0.7)
- Detection Confidence: Minimum confidence for detection (default: 0.7)

## Project Structure

```
FaceRecognition/
├── app.py                 # Main Streamlit application
├── config.py              # Configuration
├── requirements.txt       # Dependencies
├── README.md
├── models/
│   ├── facenet_model.py   # FaceNet model (keras-facenet)
│   └── face_detector.py   # MTCNN face detection
├── core/
│   ├── database_manager.py
│   └── recognition_engine.py
└── database/              # Auto-generated
```

## Model

- **Package**: keras-facenet
- **Input**: 160x160x3 RGB images
- **Output**: 512-D L2-normalized embeddings
- **Detection**: MTCNN
