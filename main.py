from squash import player
from squash import ball
from squash import court
from squash import ball

import cv2
from ultralytics import YOLO
import os
'''
players={}
video_folder = 'full-games'
posemodel=YOLO('models/yolo11s-pose.pt')
ballmodel=YOLO('squash/train7/best.pt')
conf=0.9
squashmodel=YOLO('best.pt')

        cap=cv2.VideoCapture(path)
        while cap.isOpened():
            success, frame=cap.read()
            if success:
                results=posemodel(frame, conf=conf)
                while(results[0].boxes is not None):
                    people=0
                    for class_id in results[0].boxes.cls:
                        if class_id==0:
                            people+=1
                    if people<2:
                        conf*=0.9
                        results=posemodel(frame, conf=conf)
                    else:
                        break
                    if(conf<0.1):
                        break
                annotated_frame=results[0].plot()
                cv2.imshow('Annotated Frame', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:    
                break
cap.release()
cv2.destroyAllWindows()
'''


import cv2
from ultralytics import YOLO

# Load models
pose_model = YOLO('models\\yolo11s-pose.pt')
ballmodel = YOLO('best.pt')

video_folder = 'main-games'
for video_file in os.listdir(video_folder):
    if video_file.endswith(".mp4"):
        path = video_folder + "/" + video_file
        cap = cv2.VideoCapture(path)

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            pose_results = pose_model(frame)
            custom_results = ballmodel(frame)

            annotated_frame = pose_results[0].plot()

            for box in custom_results[0].boxes:
                # Ensure box.xyxy returns four values
                coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
                x1, y1, x2, y2 = coords
                label = ballmodel.names[int(box.cls)]
                confidence = float(box.conf) 
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow('Annotated Frame', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()