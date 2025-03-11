import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self, frame_resizing=0.25):
        """
        Initialize the face recognition system.
        :param frame_resizing: Factor to resize frames for faster processing (default 0.25)
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_resizing = frame_resizing

    def load_encoding_images(self, images_path):
        """
        Load and encode faces from images in the specified directory.
        Skips images without faces or unreadable files.
        :param images_path: Path to the folder containing face images.
        """
        image_paths = glob.glob(os.path.join(images_path, "*.*"))
        print(f"{len(image_paths)} encoding images found.")

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                print(f"[ERROR] Could not read {img_path}. Skipping...")
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            filename = os.path.splitext(os.path.basename(img_path))[0]

            encodings = face_recognition.face_encodings(rgb_img)
            if encodings:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename)
            else:
                print(f"[WARNING] No face detected in {filename}. Skipping...")

        print(f"Encoding images loaded successfully with {len(self.known_face_encodings)} valid faces.")

    def detect_known_faces(self, frame):
        """
        Detect known faces in a given video frame.
        :param frame: Input frame from the video stream.
        :return: Tuple (face_locations, face_names)
        """
        small_frame = cv2.resize(frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)

            if matches:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        # Scale face locations back to original frame size
        face_locations = (np.array(face_locations) / self.frame_resizing).astype(int)
        return face_locations, face_names
