"""
Configuration settings for Streamlit Face Recognition System
"""

# ==================== Model Configuration ====================
MODEL_INPUT_SHAPE = (160, 160, 3)
EMBEDDING_SIZE = 512

# ==================== Detection Configuration ====================
MIN_DETECTION_CONFIDENCE = 0.7
DETECTION_SCALE_FACTOR = 0.6

# ==================== Recognition Configuration ====================
RECOGNITION_THRESHOLD = 0.7

# ==================== Page Configuration ====================
PAGE_TITLE = "Face Recognition System"
SIDEBAR_STATE = "expanded"

# Combined page config dict
PAGE_CONFIG = {
    "page_title": PAGE_TITLE,
    "layout": "wide",
    "initial_sidebar_state": SIDEBAR_STATE
}

# Legacy compatibility (for older code)
DETECTION_CONFIDENCE = MIN_DETECTION_CONFIDENCE
DETECTION_SCALE = DETECTION_SCALE_FACTOR
