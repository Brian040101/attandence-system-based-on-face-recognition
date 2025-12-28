import os
import cv2
import joblib
import numpy as np
import insightface

class RealTimeDetector:
    def __init__(self, model_dir="models"):
        self.model_path = os.path.join(model_dir, "face_classifier.pkl")
        self.clf = None
        self.le = None
        
        # Initialize InsightFace (same as others)
        print("Initializing Detector...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        
        # Load the model immediately if it exists
        self.load_model()

    def load_model(self):
        """Loads the KNN classifier and Label Encoder."""
        if os.path.exists(self.model_path):
            try:
                # We saved a dict {'model': clf, 'label_encoder': le}
                data = joblib.load(self.model_path)
                self.clf = data['model']
                self.le = data['label_encoder']
                print("Model loaded successfully!")
                return True
            except Exception as e:
                print(f"Error loading model: {e}")
                return False
        print("Model file not found.")
        return False

    def process_frame(self, frame):
        """
        Takes a frame, detects faces, recognizes them, draws boxes, 
        and returns the annotated frame.
        """
        if self.clf is None:
            # If no model, just return original frame with a warning text
            cv2.putText(frame, "MODEL NOT LOADED", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return frame, "No Model"

        # 1. Detect Faces
        faces = self.app.get(frame)
        
        detected_name = "Unknown"
        
        for face in faces:
            # 2. Get Embedding
            embedding = face.embedding.reshape(1, -1)
            
            # 3. Predict User
            # Predict returns the ID (0, 1, 2...)
            pred_id = self.clf.predict(embedding)[0]
            # Probabilities (confidence)
            probs = self.clf.predict_proba(embedding)
            confidence = np.max(probs)
            
            # Convert ID to Name (e.g., 0 -> "Justin")
            pred_name = self.le.inverse_transform([pred_id])[0]
            
            # (Optional) Threshold check: if confidence is too low, say Unknown
            if confidence < 0.5:
                pred_name = "Unknown"
            
            detected_name = pred_name # Update status for GUI

            # 4. Draw Box & Name
            bbox = face.bbox.astype(int)
            color = (0, 255, 0) if pred_name != "Unknown" else (0, 0, 255)
            
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            cv2.putText(frame, f"{pred_name} ({confidence:.2f})", (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return frame, detected_name