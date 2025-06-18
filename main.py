import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import threading
import os


class MaskDetectionApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Face Mask Detection")
        self.window.geometry("1024x600")

        # Model paths
        self.prototxt_path = "face.prototxt"
        self.caffemodel_path = "face.caffemodel"
        self.mask_model_path = "mask_detector.model"

        # Load face detection model
        if not os.path.exists(self.prototxt_path) or not os.path.exists(self.caffemodel_path):
            messagebox.showerror("Error", "Missing face detection model files!")
            return

        self.face_net = cv2.dnn.readNetFromCaffe(
            self.prototxt_path,
            self.caffemodel_path
        )

        # Load mask classification model
        try:
            self.mask_model = load_model(self.mask_model_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load mask model: {str(e)}")
            return

        # Initialize GUI components
        self.create_widgets()

        # Video Variables
        self.cap = None
        self.current_frame = None
        self.running = False
        self.thread = None
        self.stop_button = None  # Explicitly initialize

    def create_widgets(self):
        style = ttk.Style()
        style.configure("TButton", padding=6, relief="flat")

        ttk.Label(
            self.window,
            text="Face Mask Detection",
            font=("Arial", 18, "bold"),
            foreground="#2c3e50"
        ).pack(pady=20)

        self.canvas = tk.Canvas(
            self.window,
            width=640,
            height=480,
            bg="#f0f2f3"
        )
        self.canvas.pack(pady=10)

        btn_frame = ttk.Frame(self.window)
        btn_frame.pack(pady=10)

        ttk.Button(
            btn_frame,
            text="Load Image",
            command=self.load_image,
            width=15
        ).grid(row=0, column=0, padx=10)

        ttk.Button(
            btn_frame,
            text="Start Webcam",
            command=self.start_webcam,
            width=15
        ).grid(row=0, column=1, padx=10)

        self.stop_button = ttk.Button(
            btn_frame,
            text="Stop",
            command=self.stop_processing,
            width=15,
            
        )
        self.stop_button.grid(row=0, column=2, padx=10)

    def load_image(self):
        path = filedialog.askopenfilename()
        if not path:
            return

        frame = cv2.imread(path)
        if frame is None:
            messagebox.showerror("Error", "Failed to read image.")
            return

        self.process_frame(frame)
        self.show_frame(frame)

    def start_webcam(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Webcam not accessible.")
            return
        self.start_processing()

    def start_processing(self):
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open the stream.")
            return

        self.running = True
        self.thread = threading.Thread(target=self.process_video)
        self.thread.start()

        # Safely configure the stop button
        if self.stop_button:
            self.stop_button.config(state=tk.NORMAL)

    def stop_processing(self):
        if self.running:
            self.running = False
            self.cap.release()
            self.cap = None
            self.thread.join()

            # Safely configure the stop button
            if self.stop_button:
                self.stop_button.config(state=tk.DISABLED)
            self.current_frame = None
            self.canvas.delete("all")

    def process_video(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            processed_frame = self.process_frame(frame)
            self.show_frame(processed_frame)

    def process_frame(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame,
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=True
        )
        self.face_net.setInput(blob)
        detections = self.face_net.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array(
                    [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                # Validate coordinates
                if startX < 0: startX = 0
                if startY < 0: startY = 0
                if endX > frame.shape[1]: endX = frame.shape[1]
                if endY > frame.shape[0]: endY = frame.shape[0]

                w = endX - startX
                h = endY - startY

                if w <= 0 or h <= 0:
                    continue  # Skip invalid boxes

                # Extract face
                face = frame[startY:endY, startX:endX]

                # Check face is valid
                if face.size == 0:
                    continue

                # Resize to model input size
                face_resized = cv2.resize(face, (224, 224))
                face_normalized = face_resized / 255.0
                face_reshaped = np.expand_dims(face_normalized, axis=0)

                # Predict mask
                preds = self.mask_model.predict(face_reshaped)
                mask, no_mask = preds[0]
                label = "Mask" if mask > no_mask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # Draw results
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                text = f"{label} ({max(mask, no_mask) * 100:.1f}%)"
                cv2.putText(frame, text, (startX, startY - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        return frame
    def show_frame(self, frame):
        if frame is not None:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(image=img)

            self.canvas.img = img
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img)

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionApp(root)
    app.run()