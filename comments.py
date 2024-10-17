"""
    court and racket detection, can be put in any part of while cap is opened(cap.isOpened())
    
    for box in court_results[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = courtmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        #print(f'{label} {confidence:.2f} GOT COURT')

    for box in racket_results[0].boxes:
        coords = box.xyxy[0] if len(box.xyxy) == 1 else box.xyxy
        x1temp, y1temp, x2temp, y2temp = coords
        label = racketmodel.names[int(box.cls)]
        confidence = float(box.conf)
        cv2.rectangle(annotated_frame, (int(x1temp), int(y1temp)), (int(x2temp), int(y2temp)), (255, 0, 0), 2)
        cv2.putText(annotated_frame, f'{label} {confidence:.2f}', (int(x1temp), int(y1temp) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print(f'{label} {confidence:.2f} GOT RACKET')
"""

'''
refrence points
pixel_points = np.array([
    [x0, y0],  # Bottom left
    [x1, y1],  # Bottom right
    [x2, y2],  # Top right
    [x3, y3],  # Top left
    [x4, y4],  # Bottom middle
    [x5, y5],  # Right bottom of square
    [x6, y6],  # Top middle
    [x7, y7],  # Left bottom of square
    [x8, y8],  # Right top of square
    [x9, y9],  # Left top of square
    [x10, y10],  # T
    [x11, y11]   # Middle of T and top middle court
], dtype=np.float32)
# [0] is x val and [1] is y val
# refrence[0] is top left,
# refrence[1] is top right
# refrence[2] is bottom right
# refrence[3] is bottom left
# refrence[4] is left bottom of service box
# refrence[5] is right bottom of service box
# refrence[6] is T
# refrence[7] is left of service line
# refrence[8] is right of service line
# refrence[9] is left of the top line of the front court
# refrence[10] is right of the top line of the front court
# Define the reference points in real-world coordinates (court)
# These should be the actual coordinates of the reference points on the court


pixel_points_2d = pixel_points[:, :2]
real_world_points_2d = real_world_points[:, :2]

# Calculate the homography matrix
H, status = cv2.findHomography(pixel_points_2d, real_world_points_2d)

def transform_point(pixel_point, H):
    pixel_point_homogeneous = np.append(pixel_point, 1)  # Convert to homogeneous coordinates
    real_world_point_homogeneous = np.dot(H, pixel_point_homogeneous)
    real_world_point = real_world_point_homogeneous / real_world_point_homogeneous[2]  # Convert back to Cartesian coordinates
    return real_world_point[:2]  # Return only x and y coordinates

# Example usage
pixel_point = np.array([x, y])
real_world_point = transform_point(pixel_point, H)
print(f"Real-world coordinates: {real_world_point}")
'''

