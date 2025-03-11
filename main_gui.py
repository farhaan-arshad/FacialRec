import cv2
import os
import tkinter as tk
from tkinter import simpledialog, ttk
from PIL import Image, ImageTk
from simple_facerec import SimpleFacerec

# Initialize face recognition
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")  # Load existing images

# Load camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height


class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition")
        self.root.geometry("1040x700")
        self.root.configure(bg="#1e1e1e")

        self.title_label = ttk.Label(
            root, text="Live Face Recognition", font=("Arial", 20, "bold"), foreground="white", background="#1e1e1e"
        )
        self.title_label.pack(pady=10)

        self.video_label = tk.Label(root, bg="black", bd=2, relief="ridge")
        self.video_label.pack(pady=10)

        self.button_frame = ttk.Frame(root, style="TFrame")
        self.button_frame.pack(pady=10)

        # Register Button
        self.register_button = ttk.Button(self.button_frame, text="Register Face (R)", command=self.register_face, style="Custom.TButton")
        self.register_button.grid(row=0, column=0, padx=10)

        # Quit Button
        self.quit_button = ttk.Button(self.button_frame, text="Quit (Esc)", command=self.quit_app, style="Custom.TButton")
        self.quit_button.grid(row=0, column=1, padx=10)

        self.style = ttk.Style()
        self.style.configure("Custom.TButton", font=("Arial", 14), padding=10)
        self.style.configure("TFrame", background="#1e1e1e")

        # Bind keyboard events
        root.bind("<r>", lambda event: self.register_face())  # Press 'R' to register face
        root.bind("<Escape>", lambda event: self.quit_app())  # Press 'Esc' to quit

        self.update_video()

    def update_video(self):
        ret, frame = cap.read()
        if not ret:
            return

        # Resize frame to avoid zoomed-in effect
        frame = cv2.resize(frame, (640, 480))

        # Detect Faces
        face_locations, face_names = sfr.detect_known_faces(frame)

        for face_loc, name in zip(face_locations, face_names):
            y1, x2, y2, x1 = face_loc
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            # Draw a rounded rectangle around the face
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

            # Display name with background
            overlay = frame.copy()
            cv2.rectangle(overlay, (x1, y1 - 35), (x2, y1), color, -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            cv2.putText(frame, name, (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Convert frame to Tkinter-compatible format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img_tk = ImageTk.PhotoImage(image=img)

        # Update label with new image
        self.video_label.img_tk = img_tk
        self.video_label.configure(image=img_tk)

        # Schedule next update
        self.root.after(10, self.update_video)

    def register_face(self):
        name = simpledialog.askstring("Register Face", "Enter name for the new face:", parent=self.root)
        if not name:
            return

        ret, frame = cap.read()
        if ret:
            # Resize frame before saving to maintain consistency
            frame = cv2.resize(frame, (640, 480))

            # Create directory if it doesn't exist
            if not os.path.exists("images"):
                os.makedirs("images")

            # Save the image
            img_path = f"images/{name}.jpg"
            cv2.imwrite(img_path, frame)
            print(f"[INFO] {name} registered successfully!")

            # Reload face encodings
            sfr.load_encoding_images("images/")
            print("[INFO] Face database updated!")

    def quit_app(self):
        cap.release()
        cv2.destroyAllWindows()
        self.root.quit()


# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    root.mainloop()