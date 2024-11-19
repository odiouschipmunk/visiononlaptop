import pandas as pd
import numpy as np
import ast
from typing import Dict, Tuple, List

# Map keypoint indices to body parts
KEYPOINT_MAP = {
    0: "Nose",
    4: "Left Shoulder",
    5: "Right Shoulder", 
    6: "Left Elbow",
    7: "Right Elbow",
    8: "Left Wrist",
    9: "Right Wrist",
    10: "Left Hip",
    11: "Right Hip",
    12: "Left Knee",
    13: "Right Knee",
    14: "Left Ankle",
    15: "Right Ankle"
}

def parse_keypoints(keypoints_str: str) -> np.ndarray:
    """Convert keypoint string to numpy array"""
    return np.array(ast.literal_eval(keypoints_str))

def analyze_court_position(keypoints: np.ndarray) -> Tuple[str, str]:
    """Determine player's vertical and horizontal court position"""
    valid_points = keypoints[keypoints.any(axis=1)]
    if len(valid_points) == 0:
        return "unknown", "unknown"
    
    avg_y = np.mean(valid_points[:, 1])
    avg_x = np.mean(valid_points[:, 0])
    
    vert_pos = "back" if avg_y > 0.6 else "middle" if avg_y > 0.4 else "front"
    horiz_pos = "right" if avg_x > 0.6 else "middle" if avg_x > 0.4 else "left"
    
    return vert_pos, horiz_pos

def analyze_shot_dynamics(ball_pos: List[int], prev_ball_pos: List[int]) -> Dict:
    """Analyze ball movement and shot characteristics"""
    if prev_ball_pos is None:
        return {"speed": 0, "direction": "unknown"}
    
    dx = ball_pos[0] - prev_ball_pos[0]
    dy = ball_pos[1] - prev_ball_pos[1]
    speed = np.sqrt(dx**2 + dy**2)
    
    # Determine shot direction
    angle = np.arctan2(dy, dx)
    directions = {
        (-np.pi/4, np.pi/4): "straight",
        (np.pi/4, 3*np.pi/4): "lob",
        (-3*np.pi/4, -np.pi/4): "drop",
    }
    
    direction = next((k for k, v in directions.items() if v[0] <= angle <= v[1]), "cross-court")
    
    return {
        "speed": speed,
        "direction": direction
    }

def generate_frame_description(
    frame_count: int,
    p1_keypoints: np.ndarray,
    p2_keypoints: np.ndarray,
    ball_pos: List[int],
    prev_ball_pos: List[int],
    shot_type: str
) -> str:
    """Generate detailed natural language description of the frame"""
    
    # Analyze player positions
    p1_vert, p1_horiz = analyze_court_position(p1_keypoints)
    p2_vert, p2_horiz = analyze_court_position(p2_keypoints)
    
    # Analyze shot dynamics
    dynamics = analyze_shot_dynamics(ball_pos, prev_ball_pos)
    
    # Generate description
    description = f"Frame {frame_count}:\n"
    
    # Player 1 description
    description += f"Player 1 is positioned in the {p1_vert} {p1_horiz} of the court. "
    
    # Calculate player 1's stance and positioning
    p1_valid_points = p1_keypoints[p1_keypoints.any(axis=1)]
    if len(p1_valid_points) > 0:
        shoulder_width = np.linalg.norm(p1_keypoints[4] - p1_keypoints[5])
        description += f"Their stance is {'wide' if shoulder_width > 0.2 else 'narrow'}. "
    
    # Player 2 description
    description += f"\nPlayer 2 is positioned in the {p2_vert} {p2_horiz} of the court. "
    
    # Shot description
    if dynamics["speed"] > 0:
        description += f"\nThe ball is moving at {dynamics['speed']:.1f} pixels per frame "
        description += f"in a {dynamics['direction']} trajectory. "
    
    description += f"\nShot type: {shot_type}"
    
    return description

def analyze_csv(csv_file: str):
    """Main function to analyze CSV data and generate descriptions"""
    df = pd.read_csv(csv_file)
    prev_ball_pos = None
    
    for _, row in df.iterrows():
        p1_keypoints = parse_keypoints(row['Player 1 Keypoints'])
        p2_keypoints = parse_keypoints(row['Player 2 Keypoints'])
        ball_pos = ast.literal_eval(row['Ball Position'])
        
        description = generate_frame_description(
            row['Frame count'],
            p1_keypoints,
            p2_keypoints,
            ball_pos,
            prev_ball_pos,
            row['Shot Type']
        )
        
        print(description)
        print("-" * 80)
        
        prev_ball_pos = ball_pos

if __name__ == "__main__":
    analyze_csv("squash_game.csv")