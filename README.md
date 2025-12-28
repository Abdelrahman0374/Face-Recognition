# Face Recognition System

A real-time face recognition web application built with Streamlit, MTCNN, and FaceNet.

## Features

### Core Functionality
- **Video Recognition** - Continuous real-time face detection and recognition from webcam
- **Image Recognition** - Upload an image or take an image for recognition
- **Add Person** - Register new faces with automatic face selection (highest confidence)
- **Database Management** - View, manage, and remove registered faces

### UI & Performance
- **Button-style interface** - Consistent, modern UI across all modes
- **Automatic face selection** - No manual selection needed for registration
- **Centered displays** - Professional layout with centered video/camera feeds
- **Memory optimized** - Automatic image resizing for large uploads
- **Color-coded detection** - Green for known persons, red for unknown

## Quick Start

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

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
3. View recognition results with confidence scores

### Add a New Person
1. Select **Add Person**
2. Enter the person's name (e.g., Abdelrahman Sayed)
3. Choose your input method:
   - **Upload an Image** - Select an image from your computer
   - **Take a Photo** - Capture with your camera
4. The system automatically selects the primary face (highest confidence)
5. Click **Add to Database**

> **Best Practice**: Upload or capture an image that contains **only one person** for accurate registration. While the system can handle multiple faces and will automatically select the most prominent one (highest confidence), single-person images ensure the correct individual is registered.

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
- **Image Processing**: Pillow (PIL)

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

## Performance Optimizations

- **Frame skipping** - Process every 3rd frame in video mode
- **Batch processing** - Generate embeddings in batches
- **Image resizing** - Auto-resize large images (max 1200px) to prevent memory errors
- **Detection caching** - Reuse detection boxes for smooth video display

## UI Features

- **Dark theme** - Professional dark mode interface
- **Responsive layout** - Adapts to different screen sizes
- **Centered displays** - Video and camera feeds centered for better UX
- **Button-style controls** - Consistent primary/secondary button states
- **Real-time feedback** - Live confidence scores and detection counts

## Color Scheme

| Element | Color | Usage |
|---------|-------|-------|
| Known Person | Green (#00FF00) | Detection box & label |
| Unknown Person | Red (#0000FF) | Detection box & label |
| Primary Button | Blue (#2563eb) | Active selection |
| Hover State | Blue (#3b82f6) | Button hover |

## Requirements

See `requirements.txt` for full dependency list. Key dependencies:
- `streamlit>=1.28.0`
- `tensorflow>=2.10.0`
- `keras-facenet>=0.1.1`
- `opencv-python>=4.8.0`
- `mtcnn>=0.1.1`
- `numpy>=1.23.5`
- `pillow>=9.5.0`

## License

This project is open source and available for educational purposes.
