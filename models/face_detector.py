"""
Face Detection Module using MTCNN
Detects faces in images and video frames
"""

import cv2
import numpy as np
from mtcnn import MTCNN


class FaceDetector:
    """
    Face detection using MTCNN
    """
    def __init__(self, min_confidence=0.9, scale_factor=1.0):
        """
        Initialize face detector

        Args:
            min_confidence: Minimum detection confidence (0-1)
            scale_factor: Image scaling for faster detection (0.5 = half size)
        """
        self.min_confidence = min_confidence
        self.scale_factor = scale_factor
        self.detector = MTCNN()

    def detect_faces(self, frame):
        """
        Detect faces in a frame

        Args:
            frame: BGR image from OpenCV

        Returns:
            List of dictionaries with 'box', 'confidence', 'keypoints'
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize for faster detection
        if self.scale_factor != 1.0:
            small_frame = cv2.resize(rgb_frame, None,
                                    fx=self.scale_factor,
                                    fy=self.scale_factor)
            detections = self.detector.detect_faces(small_frame)

            # Scale boxes back to original size
            for detection in detections:
                box = detection['box']
                detection['box'] = [
                    int(box[0] / self.scale_factor),
                    int(box[1] / self.scale_factor),
                    int(box[2] / self.scale_factor),
                    int(box[3] / self.scale_factor)
                ]

                # Scale keypoints if present
                if 'keypoints' in detection:
                    for key, point in detection['keypoints'].items():
                        detection['keypoints'][key] = (
                            int(point[0] / self.scale_factor),
                            int(point[1] / self.scale_factor)
                        )
        else:
            detections = self.detector.detect_faces(rgb_frame)

        # Filter by confidence
        filtered = [d for d in detections
                   if d['confidence'] >= self.min_confidence]

        return filtered

    def detect_face(self, frame):
        """
        Detect the primary face in a frame (highest confidence)

        This method is designed for scenarios where we expect a single person
        to be the primary subject (e.g., registration, ID verification).

        Args:
            frame: BGR image from OpenCV

        Returns:
            Single detection dictionary with 'box', 'confidence', 'keypoints'
            or None if no face is detected
        """
        detections = self.detect_faces(frame)

        if not detections:
            return None

        # Return the face with highest confidence
        primary_face = max(detections, key=lambda d: d['confidence'])
        return primary_face

    def extract_face(self, frame, box, margin=20, target_size=(160, 160)):
        """
        Extract and align face from frame

        Args:
            frame: Original frame
            box: [x, y, width, height]
            margin: Pixels to add around face
            target_size: Output size for FaceNet (160, 160)

        Returns:
            Aligned face image (160x160x3)
        """
        x, y, w, h = box

        # Add margin
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)

        # Extract face
        face = frame[y1:y2, x1:x2]

        # Check if face is valid
        if face.size == 0:
            return None

        # Resize to target size
        try:
            face_resized = cv2.resize(face, target_size)
            return face_resized
        except Exception:
            return None

    def draw_detections(self, frame, detections, names=None, confidences=None):
        """
        Draw detection boxes on frame

        Args:
            frame: Frame to draw on
            detections: List of detections
            names: Optional list of names for each detection
            confidences: Optional list of confidence scores

        Returns:
            Annotated frame
        """
        output = frame.copy()

        for i, detection in enumerate(detections):
            x, y, w, h = detection['box']

            # Get name and confidence
            name = names[i] if names and i < len(names) else "Unknown"
            conf = confidences[i] if confidences and i < len(confidences) else detection['confidence']

            # Choose color based on recognition
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            # Draw label - name only
            label = f"{name}"
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Label background
            cv2.rectangle(output, (x, y - label_h - 10),
                         (x + label_w + 10, y), color, -1)

            # Label text
            cv2.putText(output, label, (x + 5, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output
