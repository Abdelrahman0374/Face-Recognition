"""
FaceNet Model Wrapper using keras-facenet package
Loads and manages the FaceNet model for face embeddings
"""

from keras_facenet import FaceNet
import numpy as np
import cv2


class FaceNetModel:
    """
    Wrapper class for FaceNet model using keras-facenet package
    """
    def __init__(self, model_path=None):
        """
        Initialize FaceNet model

        Args:
            model_path: Not used - keras-facenet handles model loading
        """
        # Load model using keras-facenet package
        self.facenet = FaceNet()
        self.model = self.facenet.model

        # Expected input size for keras-facenet model
        self.input_size = (160, 160)
        self.embedding_size = 512

    def get_embedding(self, face_image):
        """
        Generate embedding for a single face image

        Args:
            face_image: Preprocessed face image (160x160x3) in BGR format

        Returns:
            128-dimensional embedding vector (L2-normalized)
        """
        # Add batch dimension if needed
        if len(face_image.shape) == 3:
            face_image = np.expand_dims(face_image, axis=0)

        # Ensure correct size
        if face_image.shape[1:3] != (160, 160):
            face_image = np.array([cv2.resize(img, (160, 160)) for img in face_image])

        # Convert BGR to RGB (OpenCV uses BGR)
        face_image = face_image[..., ::-1].copy()

        # Get embeddings using keras-facenet
        embeddings = self.facenet.embeddings(face_image)
        embedding = embeddings[0]

        # L2 normalize
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding

    def get_embeddings_batch(self, face_images):
        """
        Generate embeddings for multiple face images

        Args:
            face_images: List or array of preprocessed face images (160x160x3)

        Returns:
            Array of 128-D embeddings (L2-normalized)
        """
        # Convert to numpy array if needed
        if isinstance(face_images, list):
            face_images = np.array(face_images)

        # Ensure correct size
        if face_images.shape[1:3] != (160, 160):
            face_images = np.array([cv2.resize(img, (160, 160)) for img in face_images])

        # Convert BGR to RGB
        face_images = face_images[..., ::-1].copy()

        # Get embeddings using keras-facenet
        embeddings = self.facenet.embeddings(face_images)

        # L2 normalize each embedding
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-10)

        return embeddings
