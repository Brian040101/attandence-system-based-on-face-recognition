import customtkinter as ctk
import threading
import cv2
from PIL import Image, ImageTk

# Import your modules
from modules.data_collector import DataCollector
from modules.trainer import ModelTrainer
from modules.detector import RealTimeDetector

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class AISuiteApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face ID Attendance System")
        self.geometry("1100x700")

        # Initialize Backend Logic
        self.collector = DataCollector()
        self.trainer = ModelTrainer()
        self.detector = RealTimeDetector()

        # Layout
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._init_sidebar()
        self._init_pages()
        
        self.show_collect()

    def _init_sidebar(self):
        self.sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        
        self.logo = ctk.CTkLabel(self.sidebar, text="FACE ID SYSTEM", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=20)

        self.btn_collect = ctk.CTkButton(self.sidebar, text="1. Collect Faces", command=self.show_collect)
        self.btn_collect.grid(row=1, column=0, padx=20, pady=10)

        self.btn_train = ctk.CTkButton(self.sidebar, text="2. Train Model", command=self.show_train)
        self.btn_train.grid(row=2, column=0, padx=20, pady=10)

        self.btn_detect = ctk.CTkButton(self.sidebar, text="3. Live Recognition", command=self.show_detect)
        self.btn_detect.grid(row=3, column=0, padx=20, pady=10)

    def _init_pages(self):
        self.frame_collect = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_train = ctk.CTkFrame(self, fg_color="transparent")
        self.frame_detect = ctk.CTkFrame(self, fg_color="transparent")
        
        self._setup_collect_ui()
        self._setup_train_ui()
        self._setup_detect_ui()

    def hide_all(self):
        # Stop any active cameras before switching pages to avoid conflicts
        self.stop_all_cameras()
        self.frame_collect.grid_forget()
        self.frame_train.grid_forget()
        self.frame_detect.grid_forget()

    def stop_all_cameras(self):
        # Helper to ensure camera releases
        if self.is_collect_cam_on: self.toggle_collect_camera()
        if self.is_detect_cam_on: self.toggle_detect_camera()

    def show_collect(self):
        self.hide_all()
        self.frame_collect.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    def show_train(self):
        self.hide_all()
        self.frame_train.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    def show_detect(self):
        self.hide_all()
        # Reload model just in case we just finished training
        self.detector.load_model()
        self.frame_detect.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

    # ==========================
    # 1. COLLECTION UI
    # ==========================
    def _setup_collect_ui(self):
        self.is_collect_cam_on = False
        
        title = ctk.CTkLabel(self.frame_collect, text="Step 1: Data Collection", font=("Arial", 24, "bold"))
        title.pack(pady=10)

        self.cam_view_collect = ctk.CTkLabel(self.frame_collect, text="Camera Off", width=480, height=360, fg_color="gray")
        self.cam_view_collect.pack(pady=10)

        self.class_entry = ctk.CTkEntry(self.frame_collect, placeholder_text="Enter Name (e.g. 'Elon')")
        self.class_entry.pack(pady=5)

        self.btn_capture = ctk.CTkButton(self.frame_collect, text="Capture Photo", command=self.capture_image)
        self.btn_capture.pack(pady=5)

        self.btn_cam_collect = ctk.CTkButton(self.frame_collect, text="Start Camera", command=self.toggle_collect_camera)
        self.btn_cam_collect.pack(pady=10)
        
        self.collect_log = ctk.CTkLabel(self.frame_collect, text="Ready...")
        self.collect_log.pack(pady=5)

    def toggle_collect_camera(self):
        if not self.is_collect_cam_on:
            if self.collector.start_camera():
                self.is_collect_cam_on = True
                self.btn_cam_collect.configure(text="Stop Camera", fg_color="red")
                self.update_collect_feed()
        else:
            self.is_collect_cam_on = False
            self.collector.stop_camera()
            self.btn_cam_collect.configure(text="Start Camera", fg_color="#1f6aa5")
            self.cam_view_collect.configure(image=None, text="Camera Off")

    def update_collect_feed(self):
        if self.is_collect_cam_on:
            ret, img_pil, raw_frame, face_data = self.collector.get_frame()
            if ret:
                self.last_frame = raw_frame
                self.last_face_data = face_data
                ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(480, 360))
                self.cam_view_collect.configure(image=ctk_img, text="")
                self.cam_view_collect.image = ctk_img
            self.after(10, self.update_collect_feed)

    def capture_image(self):
        if not self.is_collect_cam_on: return
        label = self.class_entry.get()
        if not label: 
            self.collect_log.configure(text="Enter a name!", text_color="red")
            return
        
        # Use the LAST DETECTED FACE DATA
        if not hasattr(self, 'last_face_data') or self.last_face_data is None:
             self.collect_log.configure(text="No face detected!", text_color="orange")
             return

        path = self.collector.save_snapshot(self.last_frame, self.last_face_data, label)
        if path:
            self.collect_log.configure(text=f"Saved: {label}", text_color="green")

    # ==========================
    # 2. TRAINING UI
    # ==========================
    def _setup_train_ui(self):
        title = ctk.CTkLabel(self.frame_train, text="Step 2: Model Training", font=("Arial", 24))
        title.pack(pady=20)

        btn = ctk.CTkButton(self.frame_train, text="Start Training", fg_color="green", height=50, command=self.run_train)
        btn.pack(pady=20)
        
        self.train_log = ctk.CTkTextbox(self.frame_train, width=500, height=300)
        self.train_log.pack(pady=10)

    def run_train(self):
        self.train_log.delete("1.0", "end")
        self.train_log.insert("end", "Training started... please wait.\n")
        
        def thread_target():
            result = self.trainer.train_model()
            self.train_log.insert("end", result + "\n")
            # Auto-reload detector
            self.detector.load_model()
            
        threading.Thread(target=thread_target).start()

    # ==========================
    # 3. DETECTION UI (UPDATED)
    # ==========================
    def _setup_detect_ui(self):
        self.is_detect_cam_on = False
        
        title = ctk.CTkLabel(self.frame_detect, text="Step 3: Live Recognition", font=("Arial", 24))
        title.pack(pady=10)

        # Video Display
        self.cam_view_detect = ctk.CTkLabel(self.frame_detect, text="Camera Off", width=480, height=360, fg_color="gray")
        self.cam_view_detect.pack(pady=10)

        # Result Label
        self.detect_res = ctk.CTkLabel(self.frame_detect, text="Detected: ---", font=("Arial", 20, "bold"))
        self.detect_res.pack(pady=10)

        # Start/Stop Button
        self.btn_cam_detect = ctk.CTkButton(self.frame_detect, text="Start Recognition", command=self.toggle_detect_camera)
        self.btn_cam_detect.pack(pady=10)

    def toggle_detect_camera(self):
        if not self.is_detect_cam_on:
            # Re-use the collector's camera logic or open a new cap
            # Here we just use the collector class to handle the hardware
            if self.collector.start_camera():
                self.is_detect_cam_on = True
                self.btn_cam_detect.configure(text="Stop Recognition", fg_color="red")
                self.update_detect_feed()
        else:
            self.is_detect_cam_on = False
            self.collector.stop_camera()
            self.btn_cam_detect.configure(text="Start Recognition", fg_color="#1f6aa5")
            self.cam_view_detect.configure(image=None, text="Camera Off")

    def update_detect_feed(self):
        if self.is_detect_cam_on:
            # We get the raw frame from collector
            # Note: We ignore the collector's built-in face rects because Detector does its own
            ret, _, raw_frame, _ = self.collector.get_frame()
            
            if ret:
                # 1. Pass frame to Detector
                annotated_frame, name = self.detector.process_frame(raw_frame)
                
                # 2. Update Label
                self.detect_res.configure(text=f"Detected: {name}")
                if name != "Unknown" and name != "No Model":
                    self.detect_res.configure(text_color="green")
                else:
                    self.detect_res.configure(text_color="red")

                # 3. Show Image
                frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(frame_rgb)
                ctk_img = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(480, 360))
                
                self.cam_view_detect.configure(image=ctk_img, text="")
                self.cam_view_detect.image = ctk_img
            
            self.after(10, self.update_detect_feed)

if __name__ == "__main__":
    app = AISuiteApp()
    app.mainloop()