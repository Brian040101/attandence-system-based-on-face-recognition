import os
import numpy as np
from facenet_pytorch import InceptionResnetV1
from PIL import Image
import torch
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 假設每個 student_data/XXX 資料夾都是一個學生
data_dir = "student_data"
X, y = [], []

for folder in os.listdir(data_dir):
    path = os.path.join(data_dir, folder)
    if os.path.isdir(path):
        label = folder  # 例如 "S001_John Doe"
        for fname in os.listdir(path):
            img = Image.open(os.path.join(path, fname)).convert("RGB").resize((160, 160))
            img_tensor = torch.from_numpy(np.array(img)).permute(2,0,1).float().div(255).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = facenet(img_tensor).cpu().numpy().flatten()
            X.append(emb)
            y.append(label)

X = np.stack(X)
y = np.array(y)
print("總樣本數：", X.shape[0])

# 編碼標籤
le = LabelEncoder()
y_enc = le.fit_transform(y)

# 訓練分類器（這裡用 SVM，也可以換 KNN、MLP...）
clf = svm.SVC(kernel='linear', probability=True)
clf.fit(X, y_enc)

# 驗證模型
pred = clf.predict(X)
print("訓練集準確率：", accuracy_score(y_enc, pred))

# 儲存模型
import joblib
joblib.dump(clf, "face_classifier.joblib")
joblib.dump(le, "label_encoder.joblib")
