import tkinter as tk
from tkinter import messagebox
import cv2
from PIL import Image, ImageTk
import os
import torch
from facenet_pytorch import MTCNN

# 設定 facenet-pytorch 的 MTCNN
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=False, device=device)

class FaceRegisterApp:
    def __init__(self, master):
        self.master = master
        self.master.title("學生人臉註冊")
        self.name_var = tk.StringVar()
        self.id_var = tk.StringVar()
        self.collecting = False
        self.collected = 0
        self.save_dir = None
        self.cap = None

        # 介面 layout
        tk.Label(master, text="學生姓名：").grid(row=0, column=0)
        tk.Entry(master, textvariable=self.name_var).grid(row=0, column=1)
        tk.Label(master, text="學生ID：").grid(row=1, column=0)
        tk.Entry(master, textvariable=self.id_var).grid(row=1, column=1)
        self.btn_start = tk.Button(master, text="開始收集人臉", command=self.start_capture)
        self.btn_start.grid(row=2, column=0, columnspan=2, pady=5)
        self.lbl_progress = tk.Label(master, text="未開始")
        self.lbl_progress.grid(row=3, column=0, columnspan=2)
        self.canvas = tk.Canvas(master, width=480, height=360)
        self.canvas.grid(row=4, column=0, columnspan=2)

    def start_capture(self):
        name = self.name_var.get().strip()
        sid = self.id_var.get().strip()
        if not name or not sid:
            messagebox.showerror("錯誤", "請輸入姓名和ID")
            return
        self.save_dir = f"student_data/{sid}_{name}"
        os.makedirs(self.save_dir, exist_ok=True)
        self.collected = 0
        self.collecting = True
        self.lbl_progress.config(text=f"收集進度: 0/30")
        self.btn_start.config(state="disabled")
        self.sid = sid
        self.name = name
        self.cap = cv2.VideoCapture(0)
        self.update_frame()

    def update_frame(self):
        if not self.collecting or self.collected >= 30:
            if self.cap is not None:
                self.cap.release()
            self.lbl_progress.config(text="收集完成！")
            self.collecting = False
            self.btn_start.config(state="normal")  # 關鍵：解鎖按鈕
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        boxes, probs = mtcnn.detect(pil_img)
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            if self.collected < 30:
                box = boxes[0]
                x1, y1, x2, y2 = [int(v) for v in box]
                face_crop = pil_img.crop((x1, y1, x2, y2)).resize((160,160))
                img_path = os.path.join(self.save_dir, f"{self.sid}_{self.collected+1:02d}.jpg")
                face_crop.save(img_path)
                self.collected += 1
                self.lbl_progress.config(text=f"收集進度: {self.collected}/30")

        img_tk = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((480,360)))
        self.canvas.create_image(0, 0, anchor="nw", image=img_tk)
        self.canvas.img_tk = img_tk
        if self.collected < 30:
            self.master.after(200, self.update_frame)
        else:
            self.collecting = False
            self.btn_start.config(state="normal")  # 這一行可重複註冊下個

# ...下方 main 同前一版...

if __name__ == "__main__":
    root = tk.Tk()
    app = FaceRegisterApp(root)
    root.mainloop()