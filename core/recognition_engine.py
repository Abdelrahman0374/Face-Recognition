"""
Recognition Engine
Core recognition logic for Streamlit - processes single frames
"""

import numpy as np
import time
from collections import deque


class RecognitionEngine:
    """
    Face recognition engine for Streamlit application
    """
    def __init__(self, facenet_model, face_detector, database, threshold=0.7):
        """
        Initialize recognition engine

        Args:
            facenet_model: FaceNet model wrapper for embeddings
            face_detector: Face detection module
            database: Face database with known embeddings
            threshold: Distance threshold for recognition
        """
        self.facenet = facenet_model
        self.detector = face_detector
        self.database = database
        self.threshold = threshold

        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.recognition_times = deque(maxlen=30)

    def process_frame(self, frame):
        """
        Process a single frame for face recognition

        Args:
            frame: BGR image from OpenCV

        Returns:
            Tuple of (annotated_frame, results)
            results: List of recognition results with format:
            [{
                'box': [x, y, w, h],
                'name': 'Person Name',
                'distance': 0.45,
                'confidence': 0.92,
                'detection_conf': 0.99
            }, ...]
        """
        start_time = time.time()

        # Step 1: Detect faces
        detect_start = time.time()
        detections = self.detector.detect_faces(frame)
        self.detection_times.append(time.time() - detect_start)

        if not detections:
            self.frame_times.append(time.time() - start_time)
            return frame, []

        # Step 2: Process each detected face
        results = []
        names = []
        confidences = []

        recog_start = time.time()

        for detection in detections:
            box = detection['box']
            face = self.detector.extract_face(frame, box)

            if face is not None:
                # Generate embedding
                embedding = self.facenet.get_embedding(face)

                # Find match in database
                name, distance = self._find_match(embedding)
                confidence = max(0, 1 - (distance / self.threshold))

                results.append({
                    'box': box,
                    'name': name,
                    'distance': distance,
                    'confidence': confidence,
                    'detection_conf': detection['confidence'],
                    'face': face
                })

                names.append(name)
                confidences.append(confidence)

        self.recognition_times.append(time.time() - recog_start)
        self.frame_times.append(time.time() - start_time)

        # Draw results on frame
        annotated_frame = self.detector.draw_detections(
            frame, detections, names, confidences
        )

        return annotated_frame, results

    def process_image(self, image):
        """
        Process a static image for face recognition

        Args:
            image: BGR image (numpy array)

        Returns:
            Tuple of (annotated_image, results)
        """
        return self.process_frame(image)

    def get_face_embedding(self, image):
        """
        Get embedding for the first face in an image

        Args:
            image: BGR image

        Returns:
            Tuple of (embedding, face_image) or (None, None)
        """
        detections = self.detector.detect_faces(image)

        if not detections:
            return None, None

        # Use first detected face
        box = detections[0]['box']
        face = self.detector.extract_face(image, box)

        if face is None:
            return None, None

        embedding = self.facenet.get_embedding(face)

        # Validate embedding
        if np.any(np.isnan(embedding)) or np.all(embedding == 0):
            return None, None

        return embedding, face

    def _find_match(self, embedding):
        """
        Find closest match in database

        Args:
            embedding: 128-D face embedding

        Returns:
            (name, distance) tuple
        """
        if not self.database:
            return "Unknown", float('inf')

        # Vectorized distance calculation
        names = list(self.database.keys())
        db_embeddings = np.array([self.database[name] for name in names])

        # Calculate all distances at once
        distances = np.linalg.norm(db_embeddings - embedding, axis=1)

        # Find minimum distance
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        identity = names[min_idx]

        # Check threshold
        if min_distance > self.threshold:
            identity = "Unknown"

        return identity, min_distance

    def update_database(self, new_database):
        """Update the database reference"""
        self.database = new_database

    def get_fps(self):
        """Calculate average FPS"""
        if not self.frame_times:
            return 0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0

    def get_metrics(self):
        """Get performance metrics"""
        return {
            'fps': self.get_fps(),
            'avg_detection_time_ms': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'avg_recognition_time_ms': np.mean(self.recognition_times) * 1000 if self.recognition_times else 0,
            'avg_total_time_ms': np.mean(self.frame_times) * 1000 if self.frame_times else 0
        }
