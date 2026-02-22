import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

def calculate_angle(a, b, c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

def main():
    # Setup MediaPipe PoseLandmarker
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False)
        
    detector = vision.PoseLandmarker.create_from_options(options)
    
    # Initialize variables
    counter = 0 
    stage = None # "up" or "down"
    
    # Setup video capture
    cap = cv2.VideoCapture(0)
    
    LEFT_SHOULDER = 11
    LEFT_ELBOW = 13
    LEFT_WRIST = 15
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
            
        # Convert the frame received from OpenCV to a MediaPipe Image object.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Calculate timestamp 
        frame_timestamp_ms = int(time.time() * 1000)
        
        # Perform pose landmarking on the provided single image.
        try:
            pose_result = detector.detect_for_video(mp_image, frame_timestamp_ms)
        except Exception as e:
            print(f"Detection error: {e}")
            continue
            
        image = frame.copy()
        
        try:
            if pose_result and pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                landmarks = pose_result.pose_landmarks[0] # First detected person
                
                # Get coordinates for the left arm
                shoulder_lm = landmarks[LEFT_SHOULDER]
                elbow_lm = landmarks[LEFT_ELBOW]
                wrist_lm = landmarks[LEFT_WRIST]
                
                shoulder = [shoulder_lm.x, shoulder_lm.y]
                elbow = [elbow_lm.x, elbow_lm.y]
                wrist = [wrist_lm.x, wrist_lm.y]
                
                # Calculate angle
                angle = calculate_angle(shoulder, elbow, wrist)
                
                # Visualize angle
                h, w, c = image.shape
                elbow_px = tuple(np.multiply(elbow, [w, h]).astype(int))
                
                cv2.putText(image, str(int(angle)), 
                            elbow_px, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                            )
                
                # Curl counter logic
                if angle > 160:
                    stage = "down"
                if angle < 30 and stage == "down":
                    stage = "up"
                    counter += 1
                                
                # Draw the arm connections
                shoulder_px = tuple(np.multiply(shoulder, [w, h]).astype(int))
                wrist_px = tuple(np.multiply(wrist, [w, h]).astype(int))
                
                cv2.line(image, shoulder_px, elbow_px, (245, 117, 66), 4)
                cv2.line(image, elbow_px, wrist_px, (245, 117, 66), 4)
                
                for px in [shoulder_px, elbow_px, wrist_px]:
                    cv2.circle(image, px, 6, (245, 66, 230), cv2.FILLED)
                    cv2.circle(image, px, 8, (255, 255, 255), 2)
                    
        except Exception as e:
            pass
            
        # Render curl counter box
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(counter), 
                    (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(stage) if stage else '-', 
                    (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
            
        cv2.imshow('Mediapipe Pose Rep Counter', image)
        
        # Exit condition
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
