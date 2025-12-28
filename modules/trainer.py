import os
import cv2
import numpy as np
import joblib
import insightface
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

class ModelTrainer:
    def __init__(self, data_root="data", model_dir="models"):
        self.data_root = data_root
        self.model_dir = model_dir
        
        # Ensure model directory exists
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
            
        # Output path
        self.model_path = os.path.join(self.model_dir, "face_classifier.pkl")

        # Initialize InsightFace (ArcFace)
        # Using 'buffalo_l' for consistency with DataCollector
        print("Initializing InsightFace for training...")
        self.app = insightface.app.FaceAnalysis(name='buffalo_l')
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def train_model(self):
        X, y = [], []
        processed_count = 0
        skipped_count = 0

        # 1. Traverse Data Directory
        if not os.path.exists(self.data_root):
            return "Error: Data folder not found."

        subfolders = os.listdir(self.data_root)
        if not subfolders:
            return "Error: No class folders found in data/."

        print("Starting feature extraction...")

        for label in subfolders:
            class_path = os.path.join(self.data_root, label)
            if not os.path.isdir(class_path):
                continue
            
            print(f"Processing class: {label}")
            
            for fname in os.listdir(class_path):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                img_path = os.path.join(class_path, fname)
                img = cv2.imread(img_path)
                if img is None:
                    skipped_count += 1
                    continue
                
                h, w = img.shape[:2]
                
                # A. Try Standard Detection First
                faces = self.app.get(img)
                
                if len(faces) > 0:
                    # Found a face normally
                    emb = faces[0].embedding
                    X.append(emb)
                    y.append(label)
                    processed_count += 1
                else:
                    # B. Manual Fallback for Small/Cropped Images (Your Custom Logic)
                    # If image is small (<=200px) and detection failed, assume it IS a face crop
                    if h <= 200 and w <= 200:
                        try:
                            # Access the recognition model directly
                            rec_model = None
                            if hasattr(self.app, 'models') and 'recognition' in self.app.models:
                                rec_model = self.app.models['recognition']
                            
                            if rec_model:
                                # Preprocess manually for ArcFace (112x112 standard)
                                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                img_resized = cv2.resize(img_rgb, (112, 112))
                                img_input = img_resized.astype(np.float32)
                                img_input = (img_input - 127.5) / 128.0
                                img_input = np.transpose(img_input, (2, 0, 1))
                                img_input = np.expand_dims(img_input, axis=0)
                                
                                # Run inference manually
                                if hasattr(rec_model, 'forward'): # If Wrapped Model
                                     emb = rec_model.forward(img_input)[0]
                                else: # If ONNX Runtime Session
                                    input_name = rec_model.get_inputs()[0].name
                                    output = rec_model.run(None, {input_name: img_input})[0]
                                    emb = output[0]

                                X.append(emb)
                                y.append(label)
                                processed_count += 1
                            else:
                                skipped_count += 1
                        except Exception as e:
                            print(f"Manual extraction error: {e}")
                            skipped_count += 1
                    else:
                        skipped_count += 1

        if len(X) == 0:
            return "Error: No valid face data found."

        X = np.stack(X)
        y = np.array(y)

        # 2. Encode Labels
        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # 3. Train KNN (Using logic from your uploaded file)
        print("Training KNN Classifier...")
        # Safety check: if you have fewer than 5 images, reduce neighbors to avoid crash
        n_neighbors = min(5, len(X)) 
        
        clf = KNeighborsClassifier(n_neighbors=n_neighbors, metric='euclidean', weights='distance')
        clf.fit(X, y_enc)

        # 4. Verify Accuracy
        pred = clf.predict(X)
        acc = accuracy_score(y_enc, pred)
        print(f"Training Accuracy: {acc:.2f}")

        # 5. Save Everything
        # We save both into one dict for simpler loading in the detector
        saved_data = {
            'model': clf,
            'label_encoder': le
        }
        
        joblib.dump(saved_data, self.model_path)
        
        return f"Training Complete.\nProcessed: {processed_count}, Skipped: {skipped_count}\nAccuracy: {acc:.2f}\nSaved to {self.model_path}"