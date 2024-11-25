import pandas as pd
import numpy as np
import re

def parse_coordinates(coord_str):
    """Convert string representation of coordinates to numpy array"""
    try:
        # Extract numbers using regex
        numbers = re.findall(r'[\d.]+', coord_str)
        # Convert to floats and reshape into pairs
        coords = np.array([float(x) for x in numbers]).reshape(-1, 2)
        return coords
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return None

def get_court_position(keypoints):
    """Determine player's position on court based on hip points (11,12)"""
    if keypoints is None or len(keypoints) < 17:
        return "unknown"
    
    # Get hip points (indices 11,12)
    hip_points = keypoints[11:13]
    # Filter zero points
    valid_points = hip_points[~np.all(hip_points == 0, axis=1)]
    
    if len(valid_points) == 0:
        return "unknown"
    
    x_avg = np.mean(valid_points[:, 0])
    y_avg = np.mean(valid_points[:, 1])
    
    x_pos = "center"
    if x_avg < 0.45: x_pos = "left"
    elif x_avg > 0.55: x_pos = "right"
        
    y_pos = "middle"
    if y_avg < 0.6: y_pos = "front"
    elif y_avg > 0.75: y_pos = "back"
    
    return f"{y_pos} {x_pos}"

def is_lunging(keypoints):
    """Detect if player is lunging based on knee and ankle positions"""
    if keypoints is None or len(keypoints) < 17:
        return False
    
    knees = keypoints[13:15]
    ankles = keypoints[15:17]
    
    valid_knees = knees[~np.all(knees == 0, axis=1)]
    valid_ankles = ankles[~np.all(ankles == 0, axis=1)]
    
    if len(valid_knees) == 0 or len(valid_ankles) == 0:
        return False
    
    knee_y = np.mean(valid_knees[:, 1])
    ankle_y = np.mean(valid_ankles[:, 1])
    
    return abs(knee_y - ankle_y) > 0.15

def analyze_frame(row):
    """Generate description for a single frame"""
    frame = row['Frame count']
    p1_keypoints = parse_coordinates(row['Player 1 Keypoints'])
    p2_keypoints = parse_coordinates(row['Player 2 Keypoints'])
    
    # Parse ball position - handle both string and list formats
    if isinstance(row['Ball Position'], str):
        ball_pos = [int(x) for x in re.findall(r'\d+', row['Ball Position'])]
    else:
        ball_pos = row['Ball Position']
    
    shot_type = row['Shot Type']
    
    description = []
    
    # Player 1 analysis
    p1_pos = get_court_position(p1_keypoints)
    p1_lunging = is_lunging(p1_keypoints)
    p1_desc = f"Player 1 is in the {p1_pos} area"
    if p1_lunging:
        p1_desc += " and is lunging"
    description.append(p1_desc)
    
    # Player 2 analysis
    p2_pos = get_court_position(p2_keypoints)
    p2_lunging = is_lunging(p2_keypoints)
    p2_desc = f"Player 2 is in the {p2_pos} area"
    if p2_lunging:
        p2_desc += " and is lunging"
    description.append(p2_desc)
    
    # Ball analysis
    if ball_pos and not (ball_pos == [0, 0]):
        description.append(f"Ball position: {ball_pos}")
        if shot_type:
            description.append(f"Shot type: {shot_type}")
    
    return "\n".join(description)

def main():
    df = pd.read_csv('output/final.csv')
    for index, row in df.iterrows():
        print(f"\nFrame {row['Frame count']}:")
        print(analyze_frame(row))
        print("-" * 50)

if __name__ == "__main__":
    main()