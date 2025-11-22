import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import numpy as np
import joblib

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 載入已訓練好分類模型與標籤編碼器
clf = joblib.load("face_classifier.joblib")
le = joblib.load("label_encoder.joblib")

cap = cv2.VideoCapture(0)
print("Webcam開啟，偵測人臉即即時辨識學生，按 q 離開...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    boxes, probs = mtcnn.detect(pil_img)
    faces = mtcnn(pil_img)  # shape: (N, 3, 160, 160)

    if boxes is not None:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box]
            conf = float(probs[i])
            label = "Unknown"
            show_text = f"{label}"

            # 取得 embedding，使用分類器預測
            if faces is not None and len(faces) > i:
                with torch.no_grad():
                    emb = facenet(faces[i].unsqueeze(0).to(device))
                emb_np = emb.cpu().numpy()

                # SVM 預測
                pred_id = clf.predict(emb_np)[0]
                pred_prob = clf.predict_proba(emb_np)[0].max()
                label = le.inverse_transform([pred_id])[0]
                # 可自定義信心度門檻
                if pred_prob >= 0.6:
                    show_text = f"{label} ({pred_prob:.2f})"
                    box_color = (0,255,0)
                else:
                    show_text = f"Unknown ({pred_prob:.2f})"
                    box_color = (0,0,255)
            else:
                box_color = (0,0,255)
            # 畫框與名稱
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(frame, show_text, (x1, y2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2)
            cv2.putText(frame, f"{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

    cv2.imshow("Real-time Student Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
