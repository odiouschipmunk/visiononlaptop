import cv2
from ultralytics import YOLO
import numpy as np

# Load models
pose_model = YOLO('models\\yolo11s-pose.pt')
ballmodel = YOLO('best.pt')

# Video file path
video_file = 'Squash Farag v Hesham - Houston Open 2022 - Final Highlights.mp4'
video_folder = 'full-games'
path = 'Untitled design.mp4'

cap = cv2.VideoCapture(path)

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create a blank canvas for heatmap based on video resolution
heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)


def drawmap(lx,ly,rx,ry):

    # Update heatmap at the ankle positions
    lx = min(max(lx, 0), heatmap.shape[1] - 1)  # Bound lx to [0, width-1]
    ly = min(max(ly, 0), heatmap.shape[0] - 1)  # Bound ly to [0, height-1]
    rx = min(max(rx, 0), heatmap.shape[1] - 1)  # Bound rx to [0, width-1]
    ry = min(max(ry, 0), heatmap.shape[0] - 1)
    heatmap[ly, lx] += 1
    heatmap[ry, rx] += 1
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Pose and ball detection
    pose_results = pose_model(frame)
    ball = ballmodel(frame)

    annotated_frame = pose_results[0].plot()

    # Check if keypoints exist and are not empty
    if pose_results[0].keypoints.xyn is not None and len(pose_results[0].keypoints.xyn[0]) > 0:
        for person in pose_results[0].keypoints.xyn:
            print(f'Length of person keypoints: {len(person)}')
            print(f'Person keypoints: {person}')
            
            if len(person) >= 17:  # Ensure at least 17 keypoints are present

                left_ankle_x = int(person[16][0] * frame_width)  # Scale the X coordinate
                left_ankle_y = int(person[16][1] * frame_height)  # Scale the Y coordinate
                right_ankle_x = int(person[15][0] * frame_width)  # Scale the X coordinate
                right_ankle_y = int(person[15][1] * frame_height)  # Scale the Y coordinate
                if left_ankle_x > 0 or left_ankle_y > 0 or right_ankle_x > 0 or right_ankle_y > 0:
                    drawmap(left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y)
    else:
        print("No keypoints detected in this frame.")

    # Ball detection
    for box in ball[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1, y1, x2, y2 = coords
        label = ballmodel.names[int(box.cls)]
        confidence = float(box.conf)  # Convert tensor to float
        print(f'Ball: {label} {confidence:.2f} {x1} {y1} {x2} {y2}')
        # Draw bounding box for the ball
        cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Blur and normalize the heatmap for display
    #heatmap_blurred = cv2.GaussianBlur(heatmap, (15, 15), 0)
    heatmap_normalized = cv2.normalize(heatmap, None, 100, 255, cv2.NORM_MINMAX)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8), cv2.COLORMAP_BONE)
    
    # Save the heatmap
    cv2.imwrite('foot_placement_heatmap2.png', heatmap_colored)

    # Display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
