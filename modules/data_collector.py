import cv2
import os
import uuid
import numpy as np
from PIL import Image
from insightface.app import FaceAnalysis

class DataCollector:
    def __init__(self, data_root="data"):
        self.data_root = data_root
        self.cap = None
        
        # --- INSIGHTFACE SETUP ---
        # "buffalo_l" is a good balance of speed and accuracy. 
        # It downloads automatically on first run.
        print("Loading InsightFace model... (this may take a moment)")
        self.app = FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        self.output_size = (112, 112) # ArcFace standard size

        if not os.path.exists(self.data_root):
            os.makedirs(self.data_root)

    def start_camera(self, camera_index=0):
        if self.cap is None or not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
            if not self.cap.isOpened():
                print("Error: Could not open video device.")
                return False
        return True

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.cap = None

    def get_frame(self):
        """
        Returns: (success, image_for_gui, raw_frame, largest_face_object)
        """
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # 1. InsightFace Detection
                # We simply pass the BGR frame directly
                faces = self.app.get(frame)
                
                # 2. Draw Box & Find Largest Face
                display_frame = frame.copy()
                largest_face = None
                max_area = 0
                
                for face in faces:
                    # InsightFace returns bbox as [x1, y1, x2, y2]
                    box = face.bbox.astype(int)
                    x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                    
                    # Draw rectangle
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Calculate area
                    area = (x2 - x1) * (y2 - y1)
                    if area > max_area:
                        max_area = area
                        largest_face = face # Store the whole InsightFace object

                # 3. Convert for GUI Display
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                
                return True, img_pil, frame, largest_face
                
        return False, None, None, None

    def save_snapshot(self, frame, face_data, label):
        """
        Crops the face using InsightFace bbox, aligns it (optional), and saves it.
        """
        if face_data is None:
            print("No face detected to save.")
            return None

        clean_label = "".join([c for c in label if c.isalnum() or c in (' ', '_')]).strip().replace(" ", "_")
        if not clean_label: return None

        class_dir = os.path.join(self.data_root, clean_label)
        os.makedirs(class_dir, exist_ok=True)

        # 1. Get Coordinates from InsightFace object
        bbox = face_data.bbox.astype(int)
        x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]

        # 2. Crop
        # Ensure coordinates are within image bounds
        h_img, w_img = frame.shape[:2]
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(w_img, x2); y2 = min(h_img, y2)
        
        face_img = frame[y1:y2, x1:x2]

        if face_img.size == 0:
            return None

        # 3. Resize to 112x112
        try:
            face_resized = cv2.resize(face_img, self.output_size)
        except Exception as e:
            print(f"Resize error: {e}")
            return None

        # 4. Save
        filename = f"{clean_label}_{uuid.uuid4().hex[:8]}.jpg"
        save_path = os.path.join(class_dir, filename)
        cv2.imwrite(save_path, face_resized)
        
        return save_path