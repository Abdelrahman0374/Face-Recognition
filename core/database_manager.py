"""
Database Manager
Manages the face recognition database with add/remove/update operations
"""

import pickle
import json
import os
import shutil
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime


class DatabaseManager:
    """
    Manage face recognition database
    """
    def __init__(self, db_path='database'):
        """
        Initialize database manager

        Args:
            db_path: Path to database directory
        """
        self.db_path = Path(db_path)
        self.embeddings_file = self.db_path / 'embeddings.pkl'
        self.metadata_file = self.db_path / 'metadata.json'
        self.faces_dir = self.db_path / 'faces'

        # Create directories
        self.db_path.mkdir(exist_ok=True)
        self.faces_dir.mkdir(exist_ok=True)

        # Load existing data
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()

    def _load_embeddings(self):
        """Load embeddings from file"""
        if self.embeddings_file.exists():
            try:
                with open(self.embeddings_file, 'rb') as f:
                    return pickle.load(f)
            except Exception:
                return {}
        return {}

    def _load_metadata(self):
        """Load metadata from file"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_embeddings(self):
        """Save embeddings to file"""
        try:
            with open(self.embeddings_file, 'wb') as f:
                pickle.dump(self.embeddings, f)
        except Exception:
            pass

    def _save_metadata(self):
        """Save metadata to file"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception:
            pass

    def add_person(self, name, embedding, image=None, metadata=None):
        """
        Add a person to the database

        Args:
            name: Person's name (unique identifier)
            embedding: 128-D face embedding
            image: Optional face image to save
            metadata: Optional additional information

        Returns:
            bool: True if successful
        """
        # Store embedding
        self.embeddings[name] = embedding

        # Store metadata
        self.metadata[name] = {
            'added_date': datetime.now().isoformat(),
            'num_samples': 1,
            **(metadata or {})
        }

        # Save image if provided
        if image is not None:
            person_dir = self.faces_dir / name
            person_dir.mkdir(exist_ok=True)
            image_path = person_dir / 'reference.jpg'

            cv2.imwrite(str(image_path), image)

        # Persist to disk
        self._save_embeddings()
        self._save_metadata()
        return True

    def remove_person(self, name):
        """
        Remove a person from database

        Args:
            name: Person's name

        Returns:
            bool: True if successful
        """
        if name in self.embeddings:
            del self.embeddings[name]
            if name in self.metadata:
                del self.metadata[name]

            # Remove face images
            person_dir = self.faces_dir / name
            if person_dir.exists():
                shutil.rmtree(person_dir)

            self._save_embeddings()
            self._save_metadata()
            return True
        return False

    def get_all_names(self):
        """
        Get list of all people in database

        Returns:
            List of names
        """
        return list(self.embeddings.keys())

    def get_embedding(self, name):
        """
        Get embedding for a person

        Args:
            name: Person's name

        Returns:
            128-D embedding or None
        """
        return self.embeddings.get(name)

    def get_database(self):
        """
        Get entire embeddings database

        Returns:
            Dictionary of {name: embedding}
        """
        return self.embeddings

    def get_person_image(self, name):
        """
        Get reference image for a person

        Args:
            name: Person's name

        Returns:
            Image array or None
        """
        image_path = self.faces_dir / name / 'reference.jpg'
        if image_path.exists():
            return cv2.imread(str(image_path))
        return None

    def get_stats(self):
        """
        Get database statistics

        Returns:
            Dictionary with statistics
        """
        total_size = 0
        if self.embeddings_file.exists():
            total_size = os.path.getsize(self.embeddings_file)

        return {
            'total_people': len(self.embeddings),
            'names': self.get_all_names(),
            'database_size_bytes': total_size,
            'database_size_kb': total_size / 1024
        }

    def check_name_exists(self, name):
        """
        Check if a person with this name already exists (case-insensitive)

        Args:
            name: Person's name to check

        Returns:
            bool: True if name exists, False otherwise
        """
        # Case-insensitive comparison
        existing_names = [n.lower() for n in self.embeddings.keys()]
        return name.lower() in existing_names

    def find_similar_face(self, embedding, threshold=0.7):
        """
        Check if face embedding matches any existing person

        Args:
            embedding: Face embedding to check
            threshold: Recognition threshold (lower = more strict)

        Returns:
            tuple: (is_duplicate, matched_name, distance)
                - is_duplicate: True if similar face found
                - matched_name: Name of matched person (or None)
                - distance: Distance to matched person (or None)
        """
        if not self.embeddings:
            # Empty database, no duplicates possible
            return (False, None, None)

        min_distance = float('inf')
        closest_name = None

        # Compare with all existing embeddings
        for name, stored_embedding in self.embeddings.items():
            # Calculate L2 distance
            distance = np.linalg.norm(embedding - stored_embedding)

            if distance < min_distance:
                min_distance = distance
                closest_name = name

        # Check if closest match is within threshold
        is_duplicate = min_distance < threshold

        if is_duplicate:
            return (True, closest_name, min_distance)
        else:
            return (False, None, None)

    def reload(self):
        """Reload database from disk"""
        self.embeddings = self._load_embeddings()
        self.metadata = self._load_metadata()
