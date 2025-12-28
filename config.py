"""
Configuration settings for Streamlit Face Recognition System
"""

# Model Settings (keras-facenet handles model loading automatically)
INPUT_SHAPE = (160, 160, 3)
EMBEDDING_SIZE = 512

# Detection Settings
DETECTION_CONFIDENCE = 0.7
DETECTION_SCALE = 0.6

# Recognition Settings
RECOGNITION_THRESHOLD = 0.7  # Lower = more strict

# Streamlit Settings
PAGE_TITLE = "Face Recognition System"
PAGE_ICON = "face"
SIDEBAR_STATE = "expanded"
