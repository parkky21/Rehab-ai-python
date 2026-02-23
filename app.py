import customtkinter as ctk
import tkinter as tk
import cv2
import PIL.Image
import PIL.ImageTk
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

from exercises import EXERCISES
from pipeline import (
    process_landmarks,
    EMALandmarkSmoother,
    SwayTracker,
    Session,
    ProgressionState,
    create_default_feedback_engine,
)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Setup main window
        self.title("Rehab AI - Pose Tracking & Scoring")
        self.geometry("1200x720")

        # Configure grid layout
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        # ====== Sidebar ======
        self.sidebar_frame = ctk.CTkFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame, text="Rehab AI",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 5))

        self.subtitle_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Select an exercise to\nbegin form tracking.",
            font=ctk.CTkFont(size=13),
        )
        self.subtitle_label.grid(row=1, column=0, padx=20, pady=5)

        # Exercise Selection
        self.exercise_var = ctk.StringVar(value=list(EXERCISES.keys())[0])
        self.exercise_dropdown = ctk.CTkOptionMenu(
            self.sidebar_frame,
            values=list(EXERCISES.keys()),
            variable=self.exercise_var,
            command=self.change_exercise,
        )
        self.exercise_dropdown.grid(row=2, column=0, padx=20, pady=10)

        # Start/Stop Button
        self.is_running = False
        self.start_btn = ctk.CTkButton(
            self.sidebar_frame, text="Start Analysis",
            command=self.toggle_analysis,
        )
        self.start_btn.grid(row=3, column=0, padx=20, pady=10)

        # ----- Stats Frame -----
        self.stats_frame = ctk.CTkFrame(self.sidebar_frame)
        self.stats_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

        self.reps_label = ctk.CTkLabel(
            self.stats_frame, text="REPS: 0",
            font=ctk.CTkFont(size=20, weight="bold"),
        )
        self.reps_label.pack(pady=(10, 2))

        self.stage_label = ctk.CTkLabel(
            self.stats_frame, text="STAGE: -",
            font=ctk.CTkFont(size=15),
        )
        self.stage_label.pack(pady=2)

        self.feedback_label = ctk.CTkLabel(
            self.stats_frame, text="",
            font=ctk.CTkFont(size=13, slant="italic"),
            text_color="green",
        )
        self.feedback_label.pack(pady=2)

        # ----- Scoring Frame -----
        self.score_frame = ctk.CTkFrame(self.sidebar_frame)
        self.score_frame.grid(row=5, column=0, padx=20, pady=10, sticky="ew")

        self.score_title = ctk.CTkLabel(
            self.score_frame, text="LAST REP SCORE",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.score_title.pack(pady=(8, 2))

        self.final_score_label = ctk.CTkLabel(
            self.score_frame, text="Final: --",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color="#2ECC71",
        )
        self.final_score_label.pack(pady=2)

        self.rom_score_label = ctk.CTkLabel(self.score_frame, text="ROM: --", font=ctk.CTkFont(size=12))
        self.rom_score_label.pack(pady=1)

        self.stability_score_label = ctk.CTkLabel(self.score_frame, text="Stability: --", font=ctk.CTkFont(size=12))
        self.stability_score_label.pack(pady=1)

        self.tempo_score_label = ctk.CTkLabel(self.score_frame, text="Tempo: --", font=ctk.CTkFont(size=12))
        self.tempo_score_label.pack(pady=1)

        self.session_avg_label = ctk.CTkLabel(
            self.score_frame, text="Session Avg: --",
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color="#3498DB",
        )
        self.session_avg_label.pack(pady=(4, 8))

        # ----- Feedback from engine -----
        self.engine_feedback_label = ctk.CTkLabel(
            self.sidebar_frame, text="",
            font=ctk.CTkFont(size=12),
            text_color="#E74C3C",
            wraplength=240,
        )
        self.engine_feedback_label.grid(row=6, column=0, padx=20, pady=5)

        # ====== Main Video Area ======
        self.video_frame = ctk.CTkFrame(self)
        self.video_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)

        # Use a raw tkinter Label for video to avoid CTkImage GC crash
        self.vid_label = tk.Label(
            self.video_frame, text="Camera Feed Offline",
            bg="#2b2b2b", fg="white", font=("Helvetica", 18),
        )
        self.vid_label.pack(expand=True, fill="both")

        # Camera and MediaPipe
        self.cap = None
        self.detector = None
        self.current_exercise = EXERCISES[self.exercise_var.get()]
        self.video_photo = None  # Will be a PIL.ImageTk.PhotoImage
        self.video_size = None   # (width, height) of the photo buffer

        # Pipeline components
        self.smoother = EMALandmarkSmoother(alpha=0.3)
        self.sway_tracker = SwayTracker(window_size=30)
        self.feedback_engine = create_default_feedback_engine()
        self.session = None
        self.progression = ProgressionState()
        self.progression.load("progression_state.json")

        # Initialize MediaPipe
        self.init_mediapipe()

    def init_mediapipe(self):
        try:
            base_options = python.BaseOptions(model_asset_path="pose_landmarker_lite.task")
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                output_segmentation_masks=False,
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
        except Exception as e:
            print(f"Error loading MediaPipe model: {e}")

    def change_exercise(self, choice):
        self.current_exercise = EXERCISES[choice]
        self.current_exercise.reset()
        self._reset_ui_labels()

    def _reset_ui_labels(self):
        self.reps_label.configure(text="REPS: 0")
        self.stage_label.configure(text="STAGE: -")
        self.feedback_label.configure(text="")
        self.final_score_label.configure(text="Final: --")
        self.rom_score_label.configure(text="ROM: --")
        self.stability_score_label.configure(text="Stability: --")
        self.tempo_score_label.configure(text="Tempo: --")
        self.session_avg_label.configure(text="Session Avg: --")
        self.engine_feedback_label.configure(text="")

    def toggle_analysis(self):
        if not self.is_running:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.vid_label.configure(text="Error: Cannot open camera")
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            self.is_running = True
            self.start_btn.configure(text="Stop Analysis", fg_color="red", hover_color="darkred")
            self.current_exercise.reset()
            self.smoother.reset()
            self.sway_tracker.reset()
            self.session = Session(exercise_name=self.exercise_var.get())
            self._reset_ui_labels()
            self.update_frame()
        else:
            self.is_running = False
            if self.cap:
                self.cap.release()
            self.start_btn.configure(
                text="Start Analysis",
                fg_color=["#3B8ED0", "#1F6AA5"],
                hover_color=["#36719F", "#144870"],
            )
            self.vid_label.configure(image="", text="Camera Feed Offline")
            self.video_photo = None
            self.video_size = None

            # End session and show summary
            if self.session:
                self.session.end_session()
                summary = self.session.summary()
                self.session_avg_label.configure(
                    text=f"Session Avg: {summary['avg_final_score']}"
                )
                # Save session
                ts = int(time.time())
                self.session.save(f"session_{ts}.json")

                # Progression engine
                self.progression.record_session(summary["avg_final_score"])
                prog = self.progression.compute_progression()
                self.progression.save("progression_state.json")

                if prog["action"] != "none":
                    self.engine_feedback_label.configure(
                        text=f"Progression: {prog['action'].upper()} — {prog['reason']}"
                    )

    def update_frame(self):
        if not self.is_running:
            return

        ret, frame = self.cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int(time.time() * 1000)

            annotated_image = frame.copy()

            try:
                pose_result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)

                if (
                    pose_result
                    and getattr(pose_result, "pose_landmarks", None)
                    and len(pose_result.pose_landmarks) > 0
                ):
                    raw_landmarks = pose_result.pose_landmarks[0]

                    # Pipeline Step 1: Smooth raw landmarks
                    smoothed = self.smoother.smooth(raw_landmarks)

                    # Pipeline Step 2: Compute hip center for sway tracking
                    hip_center_x = (smoothed[23].x + smoothed[24].x) / 2.0
                    current_sway = self.sway_tracker.update(hip_center_x)

                    # Pipeline Step 3: Exercise FSM processing
                    counter, stage, feedback, render_data = self.current_exercise.process(smoothed)

                    # Pipeline Step 4: Feedback engine evaluation
                    context = {
                        "sway": current_sway,
                        "current_rom": self.current_exercise.rom_tracker.current_max - self.current_exercise.rom_tracker.current_min
                        if self.current_exercise.rom_tracker.current_max > -float('inf') else 0,
                        "target_rom": self.current_exercise.config.target_rom,
                        "ideal_rep_time": self.current_exercise.config.ideal_rep_time,
                    }
                    engine_feedback = self.feedback_engine.evaluate(smoothed, context)

                    # Pipeline Step 5: If rep was just completed, record it into session
                    if self.current_exercise.rep_completed and self.current_exercise.last_rep_scores:
                        scores = self.current_exercise.last_rep_scores
                        self.session.add_rep(
                            scores=scores,
                            rom_value=self.current_exercise.rom_tracker.average_rom,
                            rep_time=self.current_exercise.tempo_tracker.average_tempo,
                            feedback=engine_feedback,
                        )

                        # Update score UI
                        self.final_score_label.configure(text=f"Final: {scores['final_score']}")
                        self.rom_score_label.configure(text=f"ROM: {scores['rom_score']}")
                        self.stability_score_label.configure(text=f"Stability: {scores['stability_score']}")
                        self.tempo_score_label.configure(text=f"Tempo: {scores['tempo_score']}")
                        self.session_avg_label.configure(
                            text=f"Session Avg: {round(self.session.avg_final_score, 1)}"
                        )

                    # Update basic labels
                    self.reps_label.configure(text=f"REPS: {counter}")
                    self.stage_label.configure(text=f"STAGE: {stage if stage else '-'}")
                    self.feedback_label.configure(text=feedback)

                    if engine_feedback:
                        self.engine_feedback_label.configure(text=engine_feedback[0])

                    # ====== Draw Skeleton ======
                    points = render_data["points"]
                    h, w, _ = annotated_image.shape

                    if render_data.get("angle", 0) != 0 and len(points) >= 2:
                        mid_px = tuple(np.multiply([points[1].x, points[1].y], [w, h]).astype(int))
                        cv2.putText(
                            annotated_image, str(int(render_data["angle"])),
                            mid_px, cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA,
                        )

                    for i in range(len(points) - 1):
                        p1 = tuple(np.multiply([points[i].x, points[i].y], [w, h]).astype(int))
                        p2 = tuple(np.multiply([points[i + 1].x, points[i + 1].y], [w, h]).astype(int))
                        cv2.line(annotated_image, p1, p2, (245, 117, 66), 4)

                    for p in points:
                        px = tuple(np.multiply([p.x, p.y], [w, h]).astype(int))
                        cv2.circle(annotated_image, px, 6, (245, 66, 230), cv2.FILLED)
                        cv2.circle(annotated_image, px, 8, (255, 255, 255), 2)

                    # Draw score overlay on video feed
                    if self.current_exercise.last_rep_scores:
                        score_text = f"Score: {self.current_exercise.last_rep_scores['final_score']}"
                        cv2.putText(
                            annotated_image, score_text, (w - 200, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA,
                        )

            except Exception as e:
                pass

            # Convert to tkinter-compatible image
            image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            pil_img = PIL.Image.fromarray(image_rgb)

            label_w = self.video_frame.winfo_width()
            label_h = self.video_frame.winfo_height()
            if label_w > 10 and label_h > 10:
                pil_img = pil_img.resize((label_w, label_h), PIL.Image.LANCZOS)

            new_size = pil_img.size
            # Reuse photo buffer via .paste() when size matches; recreate only on resize
            if self.video_photo is None or self.video_size != new_size:
                self.video_photo = PIL.ImageTk.PhotoImage(image=pil_img)
                self.video_size = new_size
                self.vid_label.configure(image=self.video_photo, text="")
            else:
                self.video_photo.paste(pil_img)

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
