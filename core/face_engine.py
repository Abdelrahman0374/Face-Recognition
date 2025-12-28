"""
Face Engine - Main Facade
Central controller for face detection, recognition, and database operations
"""

import numpy as np
import time
from collections import deque


class FaceEngine:
    """
    Main facade for face recognition system.
    All app.py interactions go through this class.
    """
    def __init__(self, facenet_model, face_detector, db_manager, threshold=0.7):
        """
        Initialize FaceEngine

        Args:
            facenet_model: FaceNet model for embeddings
            face_detector: MTCNN face detector
            db_manager: Database manager for storing faces
            threshold: Distance threshold for recognition
        """
        self.facenet = facenet_model
        self.detector = face_detector
        self.db_manager = db_manager
        self.threshold = threshold

        # Performance tracking
        self.frame_times = deque(maxlen=30)
        self.detection_times = deque(maxlen=30)
        self.recognition_times = deque(maxlen=30)

    # ==================== Face Detection ====================

    def detect_faces(self, image):
        """
        Detect all faces in an image

        Args:
            image: BGR image from OpenCV

        Returns:
            List of detections with 'box', 'confidence', 'keypoints'
        """
        return self.detector.detect_faces(image)

    def extract_face(self, image, box, margin=20):
        """
        Extract a single face from image

        Args:
            image: BGR image
            box: [x, y, w, h] bounding box
            margin: Pixels to add around face

        Returns:
            Extracted face (160x160x3) or None
        """
        return self.detector.extract_face(image, box, margin)

    def draw_detections(self, frame, detections, names=None, confidences=None):
        """
        Draw detection boxes on frame

        Args:
            frame: Frame to draw on
            detections: List of detections
            names: Optional names for each detection
            confidences: Optional confidence scores

        Returns:
            Annotated frame
        """
        return self.detector.draw_detections(frame, detections, names, confidences)

    # ==================== Recognition ====================

    def process_frame(self, frame):
        """
        Process a single frame for face recognition

        Args:
            frame: BGR image from OpenCV

        Returns:
            Tuple of (annotated_frame, results)
        """
        start_time = time.time()

        # Step 1: Detect faces
        detect_start = time.time()
        detections = self.detector.detect_faces(frame)
        self.detection_times.append(time.time() - detect_start)

        if not detections:
            self.frame_times.append(time.time() - start_time)
            return frame, []

        # Step 2: Extract all faces
        faces = []
        valid_detections = []
        for detection in detections:
            face = self.detector.extract_face(frame, detection['box'])
            if face is not None:
                faces.append(face)
                valid_detections.append(detection)

        if not faces:
            self.frame_times.append(time.time() - start_time)
            return frame, []

        # Step 3: Generate embeddings in batch
        recog_start = time.time()
        embeddings = self.facenet.get_embeddings_batch(faces)

        # Step 4: Match each embedding to database
        results = []
        names = []
        confidences = []
        database = self.db_manager.get_database()

        for i, embedding in enumerate(embeddings):
            detection = valid_detections[i]
            name, distance = self._find_match(embedding, database)
            confidence = max(0, 1 - (distance / self.threshold))

            results.append({
                'box': detection['box'],
                'name': name,
                'distance': distance,
                'confidence': confidence,
                'detection_conf': detection['confidence'],
                'face': faces[i]
            })

            names.append(name)
            confidences.append(confidence)

        self.recognition_times.append(time.time() - recog_start)
        self.frame_times.append(time.time() - start_time)

        # Draw results on frame
        annotated_frame = self.detector.draw_detections(
            frame, valid_detections, names, confidences
        )

        return annotated_frame, results

    def get_embedding(self, face):
        """
        Get embedding for an extracted face

        Args:
            face: Extracted face image (160x160x3 BGR)

        Returns:
            512-D embedding vector or None
        """
        if face is None:
            return None

        embedding = self.facenet.get_embedding(face)

        if np.any(np.isnan(embedding)) or np.all(embedding == 0):
            return None

        return embedding

    def _find_match(self, embedding, database=None):
        """
        Find closest match in database

        Args:
            embedding: Face embedding
            database: Optional database dict (uses db_manager if None)

        Returns:
            (name, distance) tuple
        """
        if database is None:
            database = self.db_manager.get_database()

        if not database:
            return "Unknown", float('inf')

        names = list(database.keys())
        db_embeddings = np.array([database[name] for name in names])

        distances = np.linalg.norm(db_embeddings - embedding, axis=1)

        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        identity = names[min_idx]

        if min_distance > self.threshold:
            identity = "Unknown"

        return identity, min_distance

    # ==================== Database Operations ====================

    def add_person(self, name, face):
        """
        Add a new person to the database

        Args:
            name: Person's name
            face: Extracted face image (160x160x3)

        Returns:
            True if successful, False otherwise
        """
        embedding = self.get_embedding(face)
        if embedding is None:
            return False

        self.db_manager.add_person(name, embedding, face)
        return True

    def remove_person(self, name):
        """
        Remove a person from the database

        Args:
            name: Person's name to remove

        Returns:
            True if successful
        """
        return self.db_manager.remove_person(name)

    def get_all_names(self):
        """Get all registered names"""
        return self.db_manager.get_all_names()

    def get_person_image(self, name):
        """Get reference image for a person"""
        return self.db_manager.get_person_image(name)

    def get_stats(self):
        """Get database statistics"""
        return self.db_manager.get_stats()

    # ==================== Metrics ====================

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
