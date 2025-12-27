"""
Streamlit Face Recognition Application
Real-time face detection and recognition with webcam support and photo upload
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.facenet_model import FaceNetModel
from models.face_detector import FaceDetector
from core.database_manager import DatabaseManager
from core.recognition_engine import RecognitionEngine
import config

# Page configuration
st.set_page_config(
    page_title=config.PAGE_TITLE,
    page_icon=config.PAGE_ICON,
    layout="wide",
    initial_sidebar_state=config.SIDEBAR_STATE
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00d4ff, #7b2cbf);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem 0;
    }
    .info-card {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
    }
    .success-message {
        color: #00ff88;
        font-weight: bold;
    }
    .warning-message {
        color: #ffaa00;
        font-weight: bold;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #00d4ff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    """Load FaceNet and Face Detector models (cached)"""
    with st.spinner("Loading FaceNet model..."):
        facenet = FaceNetModel()

    with st.spinner("Initializing face detector..."):
        detector = FaceDetector(
            min_confidence=config.DETECTION_CONFIDENCE,
            scale_factor=config.DETECTION_SCALE
        )

    return facenet, detector


def get_database_manager():
    """Get or create database manager"""
    if 'db_manager' not in st.session_state:
        db_path = os.path.join(os.path.dirname(__file__), 'database')
        st.session_state.db_manager = DatabaseManager(db_path)
    return st.session_state.db_manager


def get_recognition_engine(facenet, detector, database):
    """Get or create recognition engine"""
    if 'engine' not in st.session_state:
        st.session_state.engine = RecognitionEngine(
            facenet_model=facenet,
            face_detector=detector,
            database=database,
            threshold=config.RECOGNITION_THRESHOLD
        )
    return st.session_state.engine


def process_uploaded_image(uploaded_file):
    """Convert uploaded file to OpenCV format"""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    return image


def main():
    # Header
    st.markdown('<h1 class="main-header">Face Recognition System</h1>', unsafe_allow_html=True)

    # Load models
    try:
        facenet, detector = load_models()
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure the keras-facenet_tf23 model is in the correct location.")
        return

    # Initialize database
    db_manager = get_database_manager()
    database = db_manager.get_database()

    # Initialize recognition engine
    engine = get_recognition_engine(facenet, detector, database)

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["Live Recognition", "Upload Image", "Add Person", "Database"]
        )

        st.divider()

        # Settings
        st.subheader("Settings")
        threshold = st.slider(
            "Recognition Threshold",
            min_value=0.3,
            max_value=1.0,
            value=config.RECOGNITION_THRESHOLD,
            step=0.05,
            help="Lower = more strict matching"
        )
        engine.threshold = threshold

        detection_conf = st.slider(
            "Detection Confidence",
            min_value=0.5,
            max_value=1.0,
            value=config.DETECTION_CONFIDENCE,
            step=0.05
        )

        st.divider()

        # Database stats
        st.subheader("Database Stats")
        stats = db_manager.get_stats()
        st.metric("People in Database", stats['total_people'])
        if stats['names']:
            st.caption(f"Names: {', '.join(stats['names'])}")

    # Main content area
    if mode == "Live Recognition":
        live_recognition_mode(engine, db_manager)

    elif mode == "Upload Image":
        upload_image_mode(engine, db_manager)

    elif mode == "Add Person":
        add_person_mode(engine, db_manager, facenet, detector)

    elif mode == "Database":
        database_mode(db_manager)


def live_recognition_mode(engine, db_manager):
    """Live webcam recognition mode"""
    st.subheader("Live Face Recognition")

    col1, col2 = st.columns([3, 1])

    with col1:
        # Camera input
        camera_image = st.camera_input("Capture from webcam")

        if camera_image is not None:
            # Convert to OpenCV format
            image = Image.open(camera_image)
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Process frame
            engine.update_database(db_manager.get_database())
            annotated_frame, results = engine.process_frame(frame)

            # Convert back to RGB for display
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(display_frame, caption="Recognition Results", use_container_width=True)

            # Show results
            if results:
                st.success(f"Detected {len(results)} face(s)")
                for i, result in enumerate(results):
                    with st.expander(f"Face {i+1}: {result['name']}"):
                        st.write(f"**Confidence:** {result['confidence']:.2%}")
                        st.write(f"**Distance:** {result['distance']:.4f}")
            else:
                st.warning("No faces detected")

    with col2:
        # Metrics
        st.markdown("### Metrics")
        metrics = engine.get_metrics()
        st.metric("FPS", f"{metrics['fps']:.1f}")
        st.metric("Detection", f"{metrics['avg_detection_time_ms']:.0f}ms")
        st.metric("Recognition", f"{metrics['avg_recognition_time_ms']:.0f}ms")


def upload_image_mode(engine, db_manager):
    """Process uploaded images for recognition"""
    st.subheader("Upload Image for Recognition")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload an image containing faces to recognize"
    )

    if uploaded_file is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)

        with col2:
            st.markdown("### Recognition Results")

            # Convert to OpenCV format
            frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # Process image
            engine.update_database(db_manager.get_database())
            annotated_frame, results = engine.process_frame(frame)

            # Display results
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(display_frame, use_container_width=True)

            # Show detected faces
            if results:
                st.success(f"Found {len(results)} face(s)")
                for result in results:
                    status = "Known" if result['name'] != "Unknown" else "Unknown"
                    st.write(f"**{result['name']}** ({status}) - Confidence: {result['confidence']:.2%}")
            else:
                st.warning("No faces detected in image")


def add_person_mode(engine, db_manager, facenet, detector):
    """Add person to database via webcam or photo upload"""
    st.subheader("Add New Person")

    # Name input
    person_name = st.text_input(
        "Enter person's name",
        placeholder="e.g., John Doe",
        help="This name will be used to identify the person"
    )

    # Source selection
    add_method = st.radio(
        "Choose how to add the person",
        ["Capture from Webcam", "Upload Photo"],
        horizontal=True
    )

    face_image = None
    display_image = None

    if add_method == "Capture from Webcam":
        camera_image = st.camera_input("Capture face")

        if camera_image is not None:
            image = Image.open(camera_image)
            display_image = image
            face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    else:  # Upload Photo
        uploaded_file = st.file_uploader(
            "Upload a clear photo of the person's face",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="The photo should clearly show the person's face"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            display_image = image
            face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Preview and add
    if display_image is not None:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Preview")
            st.image(display_image, use_container_width=True)

        with col2:
            st.markdown("### Face Detection")

            # Detect faces
            detections = detector.detect_faces(face_image)

            if detections:
                # Draw detection on preview
                preview = detector.draw_detections(face_image, detections)
                preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                st.image(preview_rgb, use_container_width=True)

                st.success(f"Detected {len(detections)} face(s)")

                if len(detections) > 1:
                    st.warning("Multiple faces detected. The first face will be used.")

                # Add person button
                if person_name and st.button("Add to Database", type="primary", use_container_width=True):
                    # Get embedding
                    embedding, extracted_face = engine.get_face_embedding(face_image)

                    if embedding is not None:
                        # Add to database
                        db_manager.add_person(person_name, embedding, extracted_face)
                        engine.update_database(db_manager.get_database())

                        st.success(f"Successfully added **{person_name}** to the database!")
                        st.balloons()

                        # Clear the form
                        st.rerun()
                    else:
                        st.error("Failed to generate face embedding. Please try again with a clearer image.")

                elif not person_name:
                    st.info("Please enter a name above to add this person")
            else:
                st.error("No face detected in the image. Please try again with a clearer photo.")


def database_mode(db_manager):
    """View and manage database"""
    st.subheader("Database Management")

    stats = db_manager.get_stats()

    # Stats cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total People", stats['total_people'])
    with col2:
        st.metric("Database Size", f"{stats['database_size_kb']:.2f} KB")
    with col3:
        st.metric("Status", "Active" if stats['total_people'] > 0 else "Empty")

    st.divider()

    if stats['total_people'] > 0:
        st.markdown("### Registered People")

        # Grid layout for people
        names = db_manager.get_all_names()
        cols = st.columns(min(4, len(names)))

        for i, name in enumerate(names):
            with cols[i % len(cols)]:
                # Try to get reference image
                ref_image = db_manager.get_person_image(name)

                with st.container():
                    if ref_image is not None:
                        ref_rgb = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
                        st.image(ref_rgb, caption=name, use_container_width=True)
                    else:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            border-radius: 10px;
                            padding: 2rem;
                            text-align: center;
                            color: white;
                        ">
                            <h3>User</h3>
                            <p>{name}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Delete button
                    if st.button(f"Remove", key=f"remove_{name}"):
                        if db_manager.remove_person(name):
                            st.success(f"Removed {name}")
                            st.rerun()

        st.divider()

        # Danger zone
        with st.expander("Danger Zone"):
            st.warning("These actions cannot be undone!")

            if st.button("Clear Entire Database", type="secondary"):
                for name in names:
                    db_manager.remove_person(name)
                st.success("Database cleared!")
                st.rerun()

    else:
        st.info("The database is empty. Go to 'Add Person' to register new faces.")


if __name__ == "__main__":
    main()
