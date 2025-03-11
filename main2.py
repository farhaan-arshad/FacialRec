import os
import cv2
import face_recognition
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox, ttk
from PIL import Image, ImageTk


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1040x700")
        self.root.configure(bg="#1e1e1e")

        # Known face encodings and names
        self.known_face_encodings = []
        self.known_face_names = []
        self.load_known_faces()

        # Camera setup
        self.video_capture = cv2.VideoCapture(0)
        self.video_capture.set(3, 640)  # Width
        self.video_capture.set(4, 480)  # Height

        # UI Setup
        self.setup_ui()

        # Start video stream
        self.update_video()

    def setup_ui(self):
        # Title
        self.title_label = ttk.Label(
            self.root,
            text="🔍 Advanced Face Recognition",
            font=("Arial", 20, "bold"),
            foreground="white",
            background="#1e1e1e"
        )
        self.title_label.pack(pady=10)

        # Video Display
        self.video_label = tk.Label(self.root, bg="black", bd=2, relief="ridge")
        self.video_label.pack(pady=10)

        # Confidence Display
        self.confidence_label = ttk.Label(
            self.root,
            text="Confidence: N/A",
            font=("Arial", 12),
            foreground="white",
            background="#1e1e1e"
        )
        self.confidence_label.pack(pady=5)

        # Button Frame
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        # Buttons
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=10)

        register_button = ttk.Button(
            button_frame,
            text="📷 Register Face (R)",
            command=self.register_face
        )
        register_button.grid(row=0, column=0, padx=10)

        quit_button = ttk.Button(
            button_frame,
            text="❌ Quit (Esc)",
            command=self.quit_app
        )
        quit_button.grid(row=0, column=1, padx=10)

        # Keyboard Bindings
        self.root.bind("<r>", lambda event: self.register_face())
        self.root.bind("<Escape>", lambda event: self.quit_app())

    def draw_rectangle(self, img, top_left, bottom_right, color, thickness=2):
        """
        Draw a basic rectangle on the image

        Args:
            img (numpy.ndarray): Image to draw on
            top_left (tuple): Top-left coordinates (x, y)
            bottom_right (tuple): Bottom-right coordinates (x, y)
            color (tuple): BGR color tuple
            thickness (int): Line thickness
        """
        cv2.rectangle(img, top_left, bottom_right, color, thickness)

    def calculate_face_confidence(self, face_encoding, known_face_encodings, tolerance=0.5):
        """
        Calculate face recognition confidence score

        Args:
            face_encoding (numpy.ndarray): Encoding of the face to match
            known_face_encodings (list): List of known face encodings
            tolerance (float): Distance tolerance for face matching

        Returns:
            tuple: (matched_name, confidence_score)
        """
        if not known_face_encodings:
            return "Unknown", 0

        # Calculate face distances
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        # Find the best match
        best_match_index = np.argmin(face_distances)
        best_distance = face_distances[best_match_index]

        # Convert distance to confidence (lower distance = higher confidence)
        if best_distance <= tolerance:
            confidence_score = max(0, 1 - best_distance)
            matched_name = self.known_face_names[best_match_index]
            return matched_name, confidence_score * 100  # Convert to percentage

        return "Unknown", 0

    def load_known_faces(self):
        """Load previously registered faces from images directory"""
        images_dir = "images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        for filename in os.listdir(images_dir):
            if filename.endswith((".png", ".jpg", ".jpeg")):
                name = os.path.splitext(filename)[0]
                image_path = os.path.join(images_dir, filename)

                try:
                    # Load the image
                    face_image = face_recognition.load_image_file(image_path)

                    # Encode the face
                    face_encoding = face_recognition.face_encodings(face_image)

                    if face_encoding:
                        self.known_face_encodings.append(face_encoding[0])
                        self.known_face_names.append(name)
                    else:
                        print(f"No face found in image: {filename}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def update_video(self):
        """Continuously update video stream with face recognition"""
        ret, frame = self.video_capture.read()
        if not ret:
            return

        # Convert the image from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Process each detected face
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Calculate face match and confidence
            name, confidence = self.calculate_face_confidence(
                face_encoding,
                self.known_face_encodings
            )

            # Choose colors
            box_color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            text_color = (255, 255, 255)

            # Draw simple rectangle around the face
            self.draw_rectangle(frame, (left, top), (right, bottom), box_color, 2)

            # Display name and confidence
            font = cv2.FONT_HERSHEY_DUPLEX
            display_text = f"{name} ({confidence:.2f}%)" if confidence > 0 else name
            cv2.putText(frame, display_text, (left , top - 5), font, 0.5, text_color, 1)

            # Update confidence label
            confidence_text = f"Confidence: {confidence:.2f}%" if name != "Unknown" else "Confidence: N/A"
            self.confidence_label.config(text=confidence_text)

        # Convert to Tkinter format
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)

        # Update label
        self.video_label.img_tk = img_tk
        self.video_label.configure(image=img_tk)

        # Schedule next update
        self.root.after(10, self.update_video)

    def register_face(self):
        """Register a new face"""
        ret, frame = self.video_capture.read()
        if not ret:
            messagebox.showerror("Error", "Could not capture image")
            return

        name = simpledialog.askstring("Register", "Enter name for this face:")
        if not name:
            return

        # Find faces in the image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_encodings:
            messagebox.showwarning("Warning", "No face detected!")
            return

        # Save the first detected face
        face_encoding = face_encodings[0]
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)

        # Create directory if it doesn't exist
        os.makedirs("images", exist_ok=True)

        # Save the image
        cv2.imwrite(f"images/{name}.jpg", frame)
        messagebox.showinfo("Success", f"Face for {name} registered successfully!")

    def quit_app(self):
        """Cleanup and exit"""
        self.video_capture.release()
        cv2.destroyAllWindows()
        self.root.quit()


def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()