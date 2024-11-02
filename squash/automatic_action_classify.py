import numpy as np
from scipy.spatial.distance import euclidean
import json
import action_classifier
def detect_shot_type(sequence_data):
    """Automatically detect shot type from player and ball positions"""
    
    # Get middle frame for shot classification
    mid_idx = len(sequence_data) // 2
    prev_frames = sequence_data[:mid_idx]
    current_frame = sequence_data[mid_idx]
    next_frames = sequence_data[mid_idx + 1:]

    # Extract positions
    p1 = current_frame['player1_keypoints']
    ball = current_frame['ball_position']
    
    # Normalize ball position
    ball_x, ball_y = ball[0] / 1920, ball[1] / 1080
    
    # Calculate ball trajectory
    if len(next_frames) > 0:
        next_ball = next_frames[0]['ball_position']
        ball_dx = (next_ball[0] / 1920) - ball_x
        ball_dy = (next_ball[1] / 1080) - ball_y
    else:
        ball_dx, ball_dy = 0, 0

    # Get key body positions
    wrist = [p1['Left Wrist']['x'], p1['Left Wrist']['y']]
    shoulder = [p1['Left Shoulder']['x'], p1['Left Shoulder']['y']]
    hip = [p1['Left Hip']['x'], p1['Left Hip']['y']]
    
    # Calculate shot indicators
    arm_extension = euclidean(wrist, shoulder)
    player_height = euclidean(shoulder, hip)
    relative_ball_height = ball_y - shoulder[1]
    ball_to_player = euclidean([ball_x, ball_y], shoulder)
    
    # Detect shot type based on positions and movements
    if relative_ball_height < -0.2 and ball_dy > 0:  # Ball above player moving down
        shot_type = 'serve'
    elif ball_x < shoulder[0] and arm_extension > 0.3:  # Ball on left, extended arm
        shot_type = 'backhand'
    elif ball_x > shoulder[0] and arm_extension > 0.3:  # Ball on right, extended arm
        shot_type = 'forehand'
    elif ball_to_player < 0.3 and abs(ball_dy) < 0.1:  # Close to player, minimal vertical movement
        shot_type = 'volley'
    elif ball_dy > 0 and hip[1] < 0.7:  # Ball moving down, player in front
        shot_type = 'drop'
    else:
        shot_type = 'other'
        
    return shot_type

def load_and_preprocess_data(json_file, sequence_length=30):
    """Load JSON data and create sequences with automated shot detection"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    sequences = []
    labels = []
    
    for i in range(len(data) - sequence_length):
        sequence = []
        sequence_data = data[i:i + sequence_length]
        
        for frame in sequence_data:
            features = action_classifier.extract_features(
                frame['player1_keypoints'],
                frame['player2_keypoints'], 
                frame['ball_position']
            )
            sequence.append(features)
            
        # Automatically detect shot type
        shot_type = detect_shot_type(sequence_data)
        
        sequences.append(sequence)
        labels.append(shot_type)
    
    return np.array(sequences), np.array(labels)

if __name__=='__main__':
    load_and_preprocess_data('30fps1920.json')