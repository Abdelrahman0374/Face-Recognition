# Face Recognition System

A real-time face recognition web application built with Streamlit, MTCNN, and FaceNet.

## Features

- **Video Recognition** - Continuous real-time face detection and recognition from webcam
- **Image Recognition** - Upload an image or capture from webcam for recognition
- **Add Person** - Register new faces via webcam capture or photo upload (with face selection for group photos)
- **Database Management** - View, manage, and remove registered faces
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

### Video Recognition
1. Select "Video Recognition" from the sidebar
2. Click "Start Video" to begin real-time recognition
3. Faces are detected and recognized continuously
4. Click "Stop Video" to end the stream

### Image Recognition
1. Select "Image Recognition"
2. Choose a tab:
   - **Upload Image**: Select an image file from your computer
   - **Capture from Webcam**: Take a photo with your camera
3. View recognition results with confidence scores

### Add a New Person
1. Select "Add Person"
2. Enter the person's name
3. Choose either:
   - Capture from Webcam
   - Upload Photo
4. If multiple faces are detected, select the correct face
5. Click "Add to Database"

### Manage Database
1. Select "Database"
2. View all registered people with their photos
3. Remove people as needed

## Settings

Adjust in the sidebar:
- **Recognition Threshold**: Lower = more strict matching (default: 0.7)
- **Detection Confidence**: Minimum confidence for face detection (default: 0.7)

## Project Structure

```
Face-Recognition/
├── app.py                  # Main Streamlit application
├── config.py               # Configuration settings
├── requirements.txt        # Python dependencies
├── models/
│   ├── facenet_model.py    # FaceNet embedding model (keras-facenet)
│   └── face_detector.py    # MTCNN face detection
├── core/
│   ├── face_engine.py      # Main facade (central controller)
│   └── database_manager.py # Face database operations
└── database/               # Auto-generated face embeddings
```

## Architecture

```
app.py  ──────►  FaceEngine
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
   FaceDetector  FaceNet   DatabaseManager
```

## Technology Stack

- **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks)
- **Face Recognition**: FaceNet (keras-facenet)
- **Web Framework**: Streamlit
- **Computer Vision**: OpenCV
- **Deep Learning**: TensorFlow/Keras

## Model Details

| Component | Description |
|-----------|-------------|
| Input | 160x160x3 RGB images |
| Embeddings | 512-D L2-normalized vectors |
| Detection | MTCNN with confidence threshold |
| Matching | Euclidean distance with threshold |

### Recognition Formulas

**Distance (Euclidean/L2 Norm):**

$$d(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=1}^{512}(a_i - b_i)^2}$$


- If $\text{distance} > \text{threshold}$ → Person is labeled as "Unknown"
- If $\text{distance} \leq \text{threshold}$ → Person is recognized with the confidence score above

