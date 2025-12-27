"""
Configuration settings for Streamlit Face Recognition System
"""

import os

# Model Settings
# Model files (model.h5 + model.json) are in the models folder
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
INPUT_SHAPE = (160, 160, 3)
EMBEDDING_SIZE = 128

# Detection Settings
DETECTION_CONFIDENCE = 0.7
DETECTION_SCALE = 0.6
MIN_FACE_SIZE = 20
FACE_MARGIN = 20

# Recognition Settings
RECOGNITION_THRESHOLD = 0.7  # Lower = more strict
BATCH_SIZE = 2

# Database Settings
DATABASE_PATH = 'database'
EMBEDDINGS_FILE = 'embeddings.pkl'
METADATA_FILE = 'metadata.json'

# Streamlit Settings
PAGE_TITLE = "Face Recognition System"
PAGE_ICON = "face"
SIDEBAR_STATE = "expanded"

# Colors (RGB format for Streamlit/PIL)
COLOR_KNOWN = (0, 255, 0)      # Green
COLOR_UNKNOWN = (255, 0, 0)    # Red
COLOR_TEXT = (255, 255, 255)   # White
