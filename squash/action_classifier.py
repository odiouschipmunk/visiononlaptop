import numpy as np
import json
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def load_and_preprocess_data(json_file, sequence_length=30):
    """Load JSON data and create sequences"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    sequences = []
    labels = []
    
    for i in range(len(data) - sequence_length):
        sequence = []
        for frame in data[i:i + sequence_length]:
            # Extract player keypoints and ball position
            p1_keypoints = frame['player1_keypoints']
            p2_keypoints = frame['player2_keypoints']
            ball_pos = frame['ball_position']
            
            # Calculate relevant features
            features = extract_features(p1_keypoints, p2_keypoints, ball_pos)
            sequence.append(features)
        
        sequences.append(sequence)
        # Assuming the action label is stored in the middle frame
        labels.append(data[i + sequence_length//2].get('action', 'unknown'))
    
    return np.array(sequences), np.array(labels)

def extract_features(p1_keypoints, p2_keypoints, ball_pos):
    """Extract relevant features for action classification"""
    features = []
    
    # Player 1 features
    # Racquet arm angles (using shoulder, elbow, wrist)
    p1_racquet_angle = calculate_arm_angle(
        [p1_keypoints['Left Shoulder']['x'], p1_keypoints['Left Shoulder']['y']],
        [p1_keypoints['Left Elbow']['x'], p1_keypoints['Left Elbow']['y']],
        [p1_keypoints['Left Wrist']['x'], p1_keypoints['Left Wrist']['y']]
    )
    
    # Player stance (distance between feet)
    p1_stance = calculate_distance(
        [p1_keypoints['Left Ankle']['x'], p1_keypoints['Left Ankle']['y']],
        [p1_keypoints['Right Ankle']['x'], p1_keypoints['Right Ankle']['y']]
    )
    
    # Body rotation (shoulder alignment)
    p1_rotation = calculate_angle(
        [p1_keypoints['Left Shoulder']['x'], p1_keypoints['Left Shoulder']['y']],
        [p1_keypoints['Right Shoulder']['x'], p1_keypoints['Right Shoulder']['y']]
    )
    
    # Distance to ball
    p1_ball_dist = calculate_distance(
        [p1_keypoints['Left Wrist']['x'], p1_keypoints['Left Wrist']['y']],
        [ball_pos[0]/1920, ball_pos[1]/1080]  # Normalize ball coordinates
    )
    
    # Add features
    features.extend([
        p1_racquet_angle,
        p1_stance,
        p1_rotation,
        p1_ball_dist
    ])
    
    return np.array(features)

def calculate_arm_angle(shoulder, elbow, wrist):
    """Calculate angle between upper and lower arm"""
    vec1 = np.array([elbow[0] - shoulder[0], elbow[1] - shoulder[1]])
    vec2 = np.array([wrist[0] - elbow[0], wrist[1] - elbow[1]])
    
    # Avoid division by zero
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    
    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    # Ensure cos_angle is in valid range [-1, 1]
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points"""
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def calculate_angle(point1, point2):
    """Calculate angle between two points relative to horizontal"""
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]
    return np.arctan2(dy, dx)

def create_model(input_shape, num_classes):
    """Create LSTM model for action classification"""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model