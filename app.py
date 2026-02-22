import customtkinter as ctk
import cv2
import PIL.Image, PIL.ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import numpy as np
import time

from exercises import EXERCISES

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup main window
        self.title("Rehab AI - Pose Tracking")
        self.geometry("1100x650")
        
        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ====== Sidebar ======
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)
        
        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Rehab AI", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        
        self.subtitle_label = ctk.CTkLabel(self.sidebar_frame, text="Select an exercise to\nbegin form tracking.", font=ctk.CTkFont(size=14))
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=10)

        # Exercise Selection Dropdown
        # Start the default selection with the first exercise in the new dict
        self.exercise_var = ctk.StringVar(value=list(EXERCISES.keys())[0])
        self.exercise_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame, 
            values=list(EXERCISES.keys()),
            variable=self.exercise_var,
            command=self.change_exercise
        )
        self.exercise_dropdown.grid(row=2, column=0, padx=20, pady=20)
        
        # Start/Stop Button
        self.is_running = False
        self.start_btn = ctk.CTkButton(self.sidebar_frame, text="Start Analysis", command=self.toggle_analysis)
        self.start_btn.grid(row=3, column=0, padx=20, pady=10)
        
        # Stats Display Frame
        self.stats_frame = ctk.CTkFrame(self.sidebar_frame)
        self.stats_frame.grid(row=4, column=0, padx=20, pady=20, sticky="ew")
        
        self.reps_label = ctk.CTkLabel(self.stats_frame, text="REPS: 0", font=ctk.CTkFont(size=20, weight="bold"))
        self.reps_label.pack(pady=10)
        
        self.stage_label = ctk.CTkLabel(self.stats_frame, text="STAGE: -", font=ctk.CTkFont(size=16))
        self.stage_label.pack(pady=5)

        self.feedback_label = ctk.CTkLabel(self.stats_frame, text="", font=ctk.CTkFont(size=14, slant="italic"), text_color="green")
        self.feedback_label.pack(pady=5)

        # ====== Main Video Area ======
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Label to hold the video stream
        self.vid_label = ctk.CTkLabel(self.video_frame, text="Camera Feed Offline", font=ctk.CTkFont(size=20))
        self.vid_label.pack(expand=True, fill="both")
        
        # Variables for camera and MediaPipe
        self.cap = None
        self.detector = None
        self.current_exercise = EXERCISES[self.exercise_var.get()]
        self.video_photo = None
        
        # Initialize MediaPipe config
        self.init_mediapipe()

    def init_mediapipe(self):
        try:
            base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_segmentation_masks=False)
            self.detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error loading MediaPipe model: {e}")

    def change_exercise(self, choice):
        self.current_exercise = EXERCISES[choice]
        self.current_exercise.reset()
        self.reps_label.configure(text="REPS: 0")
        self.stage_label.configure(text="STAGE: -")
        self.feedback_label.configure(text="")

    def toggle_analysis(self):
        if not self.is_running:
            # Start
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.vid_label.configure(text="Error: Cannot open camera")
                return
            
            # Reduce camera resolution to lower the NORM_RECT warning impact/lag
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
            self.is_running = True
            self.start_btn.configure(text="Stop Analysis", fg_color="red", hover_color="darkred")
            self.current_exercise.reset()
            self.update_frame()
        else:
            # Stop
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.start_btn.configure(text="Start Analysis", fg_color=["#3B8ED0", "#1F6AA5"], hover_color=["#36719F", "#144870"])
            self.vid_label.configure(image=None, text="Camera Feed Offline")
            self.video_photo = None

    def update_frame(self):
        if not self.is_running:
            return
            
        ret, frame = self.cap.read()
        if ret:
            # MediaPipe tasks processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)
            
            # Create a copy for drawing
            annotated_image = frame.copy()
            
            try:
                pose_result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)
                
                if pose_result and getattr(pose_result, 'pose_landmarks', None) and len(pose_result.pose_landmarks) > 0:
                    landmarks = pose_result.pose_landmarks[0]
                    
                    # Pass landmarks to currently selected exercise
                    counter, stage, feedback, render_data = self.current_exercise.process(landmarks)
                    
                    # Update Sidebar text
                    self.reps_label.configure(text=f"REPS: {counter}")
                    self.stage_label.configure(text=f"STAGE: {stage if stage else '-'}")
                    self.feedback_label.configure(text=feedback)
                    
                    # Draw angles and skeleton
                    points = render_data["points"]
                    h, w, _ = annotated_image.shape
                    
                    # Draw angle text at the middle point
                    mid_px = tuple(np.multiply([points[1].x, points[1].y], [w, h]).astype(int))
                    cv2.putText(annotated_image, str(int(render_data["angle"])), 
                                mid_px, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    
                    # Draw lines forming the angle
                    for i in range(len(points) - 1):
                        p1 = tuple(np.multiply([points[i].x, points[i].y], [w, h]).astype(int))
                        p2 = tuple(np.multiply([points[i+1].x, points[i+1].y], [w, h]).astype(int))
                        cv2.line(annotated_image, p1, p2, (245, 117, 66), 4)

                    # Draw circles on landmarks
                    for p in points:
                        px = tuple(np.multiply([p.x, p.y], [w, h]).astype(int))
                        cv2.circle(annotated_image, px, 6, (245, 66, 230), cv2.FILLED)
                        cv2.circle(annotated_image, px, 8, (255, 255, 255), 2)

            except Exception as e:
                # Silently catch tracking errors so it doesn't crash the loop
                pass
                
            # Convert OpenCV image to Tkinter native image
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(image_rgb)
            
            # Resize image to fit label while maintaining aspect ratio
            label_w = self.video_frame.winfo_width()
            label_h = self.video_frame.winfo_height()
            
            # Only resize if label has a valid size
            if label_w > 10 and label_h > 10:
                # Maintain aspect ratio (e.g. 4:3 or 16:9)
                pil_img.thumbnail((label_w, label_h))
            # Convert to CTkImage for CustomTkinter
            if self.video_photo is None:
                self.video_photo = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            else:
                self.video_photo.configure(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                
            self.vid_label.configure(image=self.video_photo, text="")

        # Schedule next frame (delay of 15ms is ~60fps, use 30ms for ~30fps safely)
        if self.is_running:
            self.after(30, self.update_frame)
            
    def on_closing(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        try:
            self.destroy()
        except:
            pass

if __name__ == "__main__":
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()
