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

    def add_person(self, name, image):
        """
        Add a person to the database with complete validation

        Complete workflow:
        1. Detect face in image (using detector)
        2. Generate embedding (using facenet)
        3. Validate duplicate name (using database)
        4. Validate duplicate face (using database)
        5. Add to database if all validations pass

        Args:
            name: Person's name
            image: BGR image from OpenCV (can be any size)

        Returns:
            dict with keys:
                - 'success': bool - True if person was added
                - 'reason': str - Success message or error description
                - 'box': List[int] - Face bounding box [x, y, w, h] (if face detected)
        """
        # Step 1: Validate name is not empty
        if not name or not name.strip():
            return {
                'success': False,
                'reason': 'Name cannot be empty',
                'box': None
            }

        name = name.strip()

        # Step 2: Detect face using detector
        detection = self.detector.detect_face(image)

        if not detection:
            return {
                'success': False,
                'reason': 'No face detected in the image. Please try again with a clearer image.',
                'box': None
            }

        box = detection['box']
        confidence = detection['confidence']

        # Step 3: Extract face from image
        extracted_face = self.detector.extract_face(image, box)

        if extracted_face is None:
            return {
                'success': False,
                'reason': 'Failed to extract face from image',
                'box': box
            }

        # Step 4: Generate embedding using facenet
        embedding = self.facenet.get_embedding(extracted_face)

        if embedding is None:
            return {
                'success': False,
                'reason': 'Failed to generate face embedding. Please try again with a clearer image.',
                'box': box
            }

        # Step 5: Validate - Check for duplicate name
        if self.db_manager.check_name_exists(name):
            return {
                'success': False,
                'reason': f'A person with the name "{name}" already exists in the database',
                'box': box
            }

        # Step 6: Validate - Check for duplicate face
        is_duplicate, matched_name, distance = self.db_manager.find_similar_face(embedding, self.threshold)

        if is_duplicate:
            return {
                'success': False,
                'reason': f'This face is already registered as "{matched_name}"',
                'box': box
            }

        # Step 7: All validations passed - Add to database
        success = self.db_manager.add_person(name, embedding, extracted_face)

        if success:
            return {
                'success': True,
                'reason': f'{name} has been successfully added to the database',
                'box': box
            }
        else:
            return {
                'success': False,
                'reason': 'Failed to save to database',
                'box': box
            }

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
