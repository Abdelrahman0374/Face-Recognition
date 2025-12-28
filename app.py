"""
Streamlit Face Recognition Application
Real-time face detection and recognition with webcam support and image upload
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

# Only import FaceEngine - Facade pattern
from core.face_engine import FaceEngine
from core.database_manager import DatabaseManager
from models.facenet_model import FaceNetModel
from models.face_detector import FaceDetector
from config import (
    MODEL_INPUT_SHAPE,
    MIN_DETECTION_CONFIDENCE,
    DETECTION_SCALE_FACTOR,
    RECOGNITION_THRESHOLD,
    PAGE_CONFIG
)

# Page configuration
st.set_page_config(**PAGE_CONFIG)


# Custom CSS - Simple Dark Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Base */
    .stApp { background: #121212; font-family: 'Inter', sans-serif; }

    /* Header */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #ffffff;
        text-align: center;
        padding: 0.5rem 0;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #1a1a1a; border-right: 1px solid #333; }

    /* Buttons */
    .stButton > button {
        background: #2563eb;
        color: #fff !important;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: #3b82f6;
        color: #fff !important;
    }

    /* Messages */
    .stSuccess { background: #1e3a2f; border-left: 3px solid #22c55e; border-radius: 4px; }
    .stInfo { background: #1e2a3a; border-left: 3px solid #3b82f6; border-radius: 4px; }
    .stError { background: #3a1e1e; border-left: 3px solid #ef4444; border-radius: 4px; }

    /* Headings */
    h2, h3 { color: #fff !important; font-weight: 600; }

    /* Tabs */
    .stTabs [aria-selected="true"] { background: #2563eb; color: #fff !important; }

    /* Slider */
    .stSlider * {
        color: #ffffff !important;
    }

    /* Hide min & max labels */
    .stSlider [data-testid="stTickBarMin"],
    .stSlider [data-testid="stTickBarMax"] {
        display: none !important;
    }

    /* Text Input */
    .stTextInput > div > div > input { background: #1e1e1e; border-radius: 6px; color: #fff; }
    .stTextInput > div > div > input:focus { border-color: #2563eb; }

    /* Radio */
    .stRadio > div > label { color: #ccc !important; }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_engine():
    """Load and initialize FaceEngine (cached)"""
    with st.spinner("Loading FaceNet model..."):
        facenet_model = FaceNetModel()

    with st.spinner("Initializing face detector..."):
        face_detector = FaceDetector(
            min_confidence=MIN_DETECTION_CONFIDENCE,
            scale_factor=DETECTION_SCALE_FACTOR
        )

    with st.spinner("Connecting to database..."):
        db_manager = DatabaseManager()

    # Create and return FaceEngine facade
    return FaceEngine(
        facenet_model=facenet_model,
        face_detector=face_detector,
        db_manager=db_manager,
        threshold=RECOGNITION_THRESHOLD
    )

def main():
    # Header
    st.markdown('<h1 class="main-header">Face Recognition System</h1>', unsafe_allow_html=True)

    # Load FaceEngine (single point of contact)
    try:
        engine = load_engine()
    except Exception as e:
        st.error(f"Error loading engine: {e}")
        st.info("Please ensure the keras-facenet model is available.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Controls")

        # Mode selection
        mode = st.radio(
            "Select Mode",
            ["Video Recognition", "Image Recognition", "Add Person", "Database"]
        )

        # Stop video when switching away from Video Recognition
        if 'current_mode' not in st.session_state:
            st.session_state.current_mode = mode
        if st.session_state.current_mode != mode:
            st.session_state.current_mode = mode
            if 'video_running' in st.session_state:
                st.session_state.video_running = False

        st.divider()

        # Settings
        st.subheader("Settings")
        threshold = st.slider(
            "Recognition Threshold",
            min_value=0.3,
            max_value=1.0,
        value=RECOGNITION_THRESHOLD,
            step=0.05,
            help="Lower = more strict matching"
        )
        engine.threshold = threshold

        st.divider()

        # Database stats (now through engine)
        st.subheader("Database Stats")
        stats = engine.get_stats()
        st.metric("People in Database", stats['total_people'])
        if stats['names']:
            st.caption(f"Names: {', '.join(stats['names'])}")

    # Main content area - all modes only receive engine
    if mode == "Video Recognition":
        video_recognition_mode(engine)

    elif mode == "Image Recognition":
        image_recognition_mode(engine)

    elif mode == "Add Person":
        add_person_mode(engine)

    elif mode == "Database":
        database_mode(engine)

def video_recognition_mode(engine):
    """Continuous video recognition mode with live webcam stream"""
    st.subheader("Video Recognition")

    st.info("Enable the webcam to start real-time face recognition. The video will continuously process and recognize faces.")

    # Initialize session state for video
    if 'video_running' not in st.session_state:
        st.session_state.video_running = False

    # Control buttons
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Start Video", type="primary", use_container_width=True, disabled=st.session_state.video_running):
            st.session_state.video_running = True
            st.rerun()
    with btn_col2:
        if st.button("Stop Video", type="secondary", use_container_width=True, disabled=not st.session_state.video_running):
            st.session_state.video_running = False
            st.rerun()

    if st.session_state.video_running:
        # Placeholder for video frame - centered with equal spacing
        col_vid = st.columns([15, 70, 15])  # 15% left, 70% video, 15% right

        with col_vid[1]:  # Center column
            video_placeholder = st.empty()

        results_placeholder = st.empty()

        # Open webcam with optimized settings
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            st.error("Could not open webcam. Please check your camera connection.")
            st.session_state.video_running = False
        else:
            try:
                frame_count = 0
                process_every_n = 3  # Process recognition every N frames
                last_results = []
                last_detections = []  # Cache detection boxes

                while st.session_state.video_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to read frame from webcam")
                        break

                    frame_count += 1

                    # Only run heavy processing every N frames
                    if frame_count % process_every_n == 0:
                        _, results = engine.process_frame(frame)
                        last_results = results
                        # Cache detection info for drawing on subsequent frames
                        last_detections = [(r['box'], r['name'], r['confidence']) for r in results] if results else []

                    # Always draw cached detections on current live frame
                    display_frame = frame.copy()
                    for box, name, confidence in last_detections:
                        x, y, w, h = box
                        # Draw rectangle - Green for known, Red for unknown (BGR format)
                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                        # Draw label
                        label = f"{name}"
                        cv2.putText(display_frame, label, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                    # Convert BGR to RGB for display
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

                    # Update video display
                    video_placeholder.image(display_frame, channels="RGB", use_container_width=True)

                    # Update results text
                    if last_results:
                        result_text = f"**Detected {len(last_results)} face(s):** "
                        result_text += ", ".join([f"{r['name']}" for r in last_results])
                        results_placeholder.success(result_text)
                    else:
                        results_placeholder.info("No faces detected")

            except Exception as e:
                st.error(f"Video error: {e}")
            finally:
                cap.release()
    else:
        st.markdown("""
        <div style="
            background: rgba(255, 255, 255, 0.05);
            border: 2px dashed rgba(255, 255, 255, 0.2);
            border-radius: 10px;
            padding: 4rem 2rem;
            text-align: center;
        ">
            <p style="font-size: 1.2rem; color: rgba(255, 255, 255, 0.6);">
                Click "Start Video" to begin real-time face recognition
            </p>
        </div>
        """, unsafe_allow_html=True)

def image_recognition_mode(engine):
    """Process images for recognition - upload or capture from webcam"""
    st.subheader("Image Recognition")

    # Initialize session state for method selection
    if 'img_recog_method' not in st.session_state:
        st.session_state.img_recog_method = "Upload an Image"

    # Button selection for upload or capture
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Upload an Image", type="primary" if st.session_state.img_recog_method == "Upload an Image" else "secondary", use_container_width=True):
            st.session_state.img_recog_method = "Upload an Image"
            st.rerun()
    with btn_col2:
        if st.button("Take a Photo", type="primary" if st.session_state.img_recog_method == "Take a Photo" else "secondary", use_container_width=True):
            st.session_state.img_recog_method = "Take a Photo"
            st.rerun()

    st.divider()

    face_image = None
    display_image = None

    if st.session_state.img_recog_method == "Upload an Image":
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing faces to recognize"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            display_image = image
            face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


    else:  # Take a Photo
        # Center the camera input
        cam_cols = st.columns([15, 70, 15])
        with cam_cols[1]:
            camera_image = st.camera_input("Take photo")

        if camera_image is not None:
            image = Image.open(camera_image)
            display_image = image
            face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Process and display results if we have an image
    if face_image is not None:
        st.divider()

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Image")
            st.image(display_image, use_container_width=True)

        with col2:
            st.markdown("### Recognition Results")

            # Process image
            annotated_frame, results = engine.process_frame(face_image)

            # Display results
            display_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            st.image(display_frame, use_container_width=True)

            # Show detected faces
            if results:
                st.success(f"Found {len(results)} face(s)")
                for result in results:
                    st.write(f"**{result['name']}**")
            else:
                st.warning("No faces detected in image")

def add_person_mode(engine):
    """Add person to database via webcam or image upload"""
    st.subheader("Add New Person")

    # Name input
    person_name = st.text_input(
        "Enter person's name",
        placeholder="e.g., Abdelrahman Sayed",
        help="This name will be used to identify the person"
    )

    # Initialize session state for method selection
    if 'add_person_method' not in st.session_state:
        st.session_state.add_person_method = "Upload an Image"

    # Button selection for source
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Upload an Image", type="primary" if st.session_state.add_person_method == "Upload an Image" else "secondary", use_container_width=True, key="btn_upload"):
            st.session_state.add_person_method = "Upload an Image"
            st.rerun()
    with btn_col2:
        if st.button("Take a Photo", type="primary" if st.session_state.add_person_method == "Take a Photo" else "secondary", use_container_width=True, key="btn_webcam"):
            st.session_state.add_person_method = "Take a Photo"
            st.rerun()

    st.divider()

    face_image = None
    display_image = None

    # Process Inputs
    if st.session_state.add_person_method == "Take a Photo":
        # Center the camera input
        cam_cols = st.columns([15, 70, 15])
        with cam_cols[1]:
            camera_image = st.camera_input("Take photo")

        if camera_image is not None:
            image = Image.open(camera_image)
            display_image = image
            face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    else:  # Upload an Image
        uploaded_file = st.file_uploader(
            "Upload a clear image of the person's face",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="The image should clearly show the person's face"
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            display_image = image
            face_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


    # Process and add person
    if display_image is not None and person_name:
        # Preview section
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Preview")
            st.image(display_image, use_container_width=True)

        with col2:
            st.markdown("### Add to Database")

            # Resize image if too large to prevent memory errors
            max_dimension = 1200
            h, w = face_image.shape[:2]
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                face_image_resized = cv2.resize(face_image, (new_w, new_h))
                st.caption(f"Image resized from {w}x{h} to {new_w}x{new_h} for processing")
            else:
                face_image_resized = face_image

            # Add person button
            if st.button("Add to Database", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    # Call FaceEngine's comprehensive add_person method
                    result = engine.add_person(person_name, face_image_resized)

                if result['success']:
                    # Success - show detection box and success message
                    if result['box']:
                        # Scale box back to original size if image was resized
                        box = result['box']
                        if max(h, w) > max_dimension:
                            scale_back = max(h, w) / max_dimension
                            box = [int(coord * scale_back) for coord in box]

                        # Draw detection box on original image
                        preview = face_image.copy()
                        x, y, w_box, h_box = box
                        cv2.rectangle(preview, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                        cv2.putText(preview, person_name, (x, y - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                        st.image(preview_rgb, caption="Detected Face", use_container_width=True)

                    st.success(result['reason'])
                    st.balloons()
                    st.rerun()
                else:
                    # Error - show appropriate message
                    st.error(f"**Failed to Add Person**\n\n{result['reason']}")

                    # Show preview with box if available
                    if result['box']:
                        box = result['box']
                        if max(h, w) > max_dimension:
                            scale_back = max(h, w) / max_dimension
                            box = [int(coord * scale_back) for coord in box]

                        preview = face_image.copy()
                        x, y, w_box, h_box = box
                        cv2.rectangle(preview, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                        st.image(preview_rgb, caption="Detected Face", use_container_width=True)

    elif display_image is not None and not person_name:
        st.info("Please enter a name above to add this person")

def database_mode(engine):
    """View and manage database"""
    st.subheader("Database Management")

    stats = engine.get_stats()

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
        names = engine.get_all_names()
        cols = st.columns(min(4, len(names)))

        for i, name in enumerate(names):
            with cols[i % len(cols)]:
                # Try to get reference image
                ref_image = engine.get_person_image(name)

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
                        if engine.remove_person(name):
                            st.success(f"Removed {name}")
                            st.rerun()

        st.divider()

        # Danger zone
        with st.expander("Danger Zone"):
            st.warning("These actions cannot be undone!")

            if st.button("Clear Entire Database", type="secondary"):
                for name in names:
                    engine.remove_person(name)
                st.success("Database cleared!")
                st.rerun()

    else:
        st.info("The database is empty. Go to 'Add Person' to register new faces.")

if __name__ == "__main__":
    main()
