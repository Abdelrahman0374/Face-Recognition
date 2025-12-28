# Face Recognition System

A real-time face recognition web application built with Streamlit, MTCNN, and FaceNet.

## Features

### Core Functionality
- **Video Recognition** - Continuous real-time face detection and recognition from webcam
- **Image Recognition** - Upload an image or take a photo for recognition
- **Add Person** - Register new faces with automatic face selection and duplicate detection
- **Database Management** - View, manage, and remove registered faces
- **Duplicate Detection** - Automatic validation prevents duplicate names and faces

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
1. Select **Video Recognition** from the sidebar
2. Click **Start Video** to begin real-time recognition
3. Faces are detected and recognized continuously with color-coded boxes:
   - 🟢 **Green** - Known person
   - 🔴 **Red** - Unknown person
4. Click **Stop Video** to end the stream

### Image Recognition
1. Select **Image Recognition**
2. Choose your input method:
   - **Upload an Image** - Select an image file from your computer
   - **Take a Photo** - Capture with your camera
3. View recognition results with name-based identification

### Add a New Person
1. Select **Add Person**
2. Enter the person's name (e.g., Abdelrahman Sayed)
3. Choose your input method:
   - **Upload an Image** - Select an image from your computer
   - **Take a Photo** - Capture with your camera
4. The system automatically selects the primary face (highest confidence)
5. Validates that name and face aren't already registered
6. Click **Add to Database**

> **Best Practice**: Upload or capture an image that contains **only one person** for accurate registration.

### Manage Database
1. Select **Database**
2. View all registered people with their images
3. Remove people using the **Remove** button
4. Use **Clear Entire Database** to reset (in Danger Zone)

## Settings

Adjust in the sidebar:
- **Recognition Threshold** - Lower = more strict matching (default: 0.7)
  - Range: 0.3 to 1.0
  - Recommended: 0.6-0.8 for balanced accuracy

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

**Design Pattern**: Facade Pattern
- `FaceEngine` serves as a single point of contact
- Abstracts complexity of detection, recognition, and database operations
- Simplified API for the Streamlit UI

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
| Auto-Selection | Highest confidence face (for registration) |

### Recognition Process

1. **Face Detection** - MTCNN detects faces and provides confidence scores
2. **Face Extraction** - Crop and resize to 160x160
3. **Embedding Generation** - FaceNet generates 512-D vector
4. **Database Matching** - Compare with stored embeddings using L2 distance
5. **Identification** - Return best match if distance < threshold

### Recognition Formulas

**Distance (Euclidean/L2 Norm):**

$$d(a, b) = \|a - b\|_2 = \sqrt{\sum_{i=1}^{512}(a_i - b_i)^2}$$

**Recognition Decision:**
- If $\text{distance} > \text{threshold}$ → Person is labeled as **"Unknown"**
- If $\text{distance} \leq \text{threshold}$ → Person is **recognized**

## License

This project is open source and available for educational purposes.
