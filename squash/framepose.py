import norfair
from norfair import Detection, Tracker, draw_tracked_objects
import numpy as np

def framepose(
    pose_model,
    frame,
    otherTrackIds,
    updated,
    references1,
    references2,
    pixdiffs,
    players,
    frame_count,
    player_last_positions,
    frame_width,
    frame_height,
    annotated_frame,
    max_players=2,
):
    try:
        # Initialize Norfair tracker if not exists
        if not hasattr(framepose, 'tracker'):
            framepose.tracker = Tracker(
                distance_function=euclidean_distance,
                distance_threshold=30,
                hit_counter_max=10,
                initialization_delay=3
            )

        # Get YOLO predictions
        results = pose_model(frame, show=False)
        
        # Convert YOLO keypoints to Norfair detections
        detections = []
        if results and hasattr(results[0], "keypoints") and results[0].keypoints is not None:
            keypoints = results[0].keypoints.cpu().numpy()
            
            for kp in keypoints:
                # Convert keypoints to pixel coordinates
                points = []
                for point in kp.xyn[0]:
                    x = int(point[0] * frame_width)
                    y = int(point[1] * frame_height)
                    points.append([x, y])
                
                # Create Norfair detection
                detection = Detection(np.array(points))
                detections.append(detection)

        # Update tracker
        tracked_objects = framepose.tracker.update(detections=detections)

        # Process each tracked object
        for tracked_obj in tracked_objects:
            track_id = tracked_obj.id
            points = tracked_obj.estimate
            
            # Calculate bounding box from keypoints
            x_coords = points[:, 0]
            y_coords = points[:, 1]
            x = np.min(x_coords)
            y = np.min(y_coords)
            w = np.max(x_coords) - x
            h = np.max(y_coords) - y

            # Player identification logic (keeping existing logic)
            if not Functions.find_match_2d_array(otherTrackIds, track_id):
                if updated[0][1] > updated[1][1]:
                    if len(references2) > 1:
                        otherTrackIds.append([track_id, 2])
                else:
                    otherTrackIds.append([track_id, 1])

            playerid = determine_player_id(track_id, updated, otherTrackIds)
            
            # Update player data
            if playerid in players:
                players[playerid].add_pose(points)
                player_last_positions[playerid] = (x, y)
                update_player_status(playerid, updated, frame_count)
            
            # Draw keypoints and ID
            for point in points:
                x, y = point
                color = (0, 0, 255) if playerid == 1 else (255, 0, 0)
                cv2.circle(annotated_frame, (int(x), int(y)), 3, color, 5)
            
            # Draw player ID
            cv2.putText(
                annotated_frame,
                f"{playerid}",
                (int(x), int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.5,
                (255, 255, 255),
                7,
            )

            # Update references and pixel differences
            update_references(playerid, frame, [x, y, w, h], references1, references2, pixdiffs)

        return [
            pose_model,
            frame,
            otherTrackIds,
            updated,
            references1,
            references2,
            pixdiffs,
            players,
            frame_count,
            player_last_positions,
            frame_width,
            frame_height,
            annotated_frame,
        ]

    except Exception as e:
        print(f"Error in framepose: {str(e)}")
        return [
            pose_model,
            frame,
            otherTrackIds,
            updated,
            references1,
            references2,
            pixdiffs,
            players,
            frame_count,
            player_last_positions,
            frame_width,
            frame_height,
            annotated_frame,
        ]

def euclidean_distance(detection, tracked_object):
    """Calculate distance between detection and tracked object"""
    return np.linalg.norm(detection.points - tracked_object.estimate)

def determine_player_id(track_id, updated, otherTrackIds):
    """Determine player ID based on tracking info"""
    if track_id == 1:
        return 1
    elif track_id == 2:
        return 2
    elif updated[0][1] > updated[1][1]:
        return 2
    elif updated[0][1] < updated[1][1]:
        return 1
    return 1

def update_player_status(playerid, updated, frame_count):
    """Update player status"""
    if playerid == 1:
        updated[0][0] = True
        updated[0][1] = frame_count
    elif playerid == 2:
        updated[1][0] = True
        updated[1][1] = frame_count

def update_references(playerid, frame, bbox, references1, references2, pixdiffs):
    """Update reference measurements"""
    x, y, w, h = bbox
    if playerid == 1:
        references1.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
        if len(references1) > 1 and len(references2) > 1:
            if len(pixdiffs) < 5:
                pixdiffs.append(abs(references1[-1] - references2[-1]))
    elif playerid == 2:
        references2.append(Functions.sum_pixels_in_bbox(frame, [x, y, w, h]))
        if len(references1) > 1 and len(references2) > 1:
            if len(pixdiffs) < 5:
                pixdiffs.append(abs(references1[-1] - references2[-1]))