import json
import numpy as np
import tensorflow as tf
from squash.action_classifier import extract_features

def classify_actions_in_file(json_file, model_path, sequence_length=30):
    """Classify actions in a JSON file containing squash game data"""
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    # Process sequences and make predictions
    results = []
    
    for i in range(len(data) - sequence_length):
        # Extract sequence
        sequence = []
        for frame in data[i:i + sequence_length]:
            features = extract_features(
                frame['player1_keypoints'],
                frame['player2_keypoints'],
                frame['ball_position']
            )
            sequence.append(features)
        
        # Convert to numpy array and reshape for model
        sequence = np.array([sequence])  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(sequence, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        # Store result
        results.append({
            'frame_start': data[i]['frame_number'],
            'frame_end': data[i + sequence_length - 1]['frame_number'],
            'predicted_action': predicted_class,
            'confidence': float(confidence)
        })
    
    return results

def main():
    # Paths
    json_file = '30fps1920.json'
    model_path = 'squash_action_classifier.h5'
    
    # Classify actions
    results = classify_actions_in_file(json_file, model_path)
    
    # Print or save results
    print("\nAction Classifications:")
    print("----------------------")
    for r in results:
        print(f"Frames {r['frame_start']}-{r['frame_end']}: "
              f"Action {r['predicted_action']} "
              f"(Confidence: {r['confidence']:.2f})")
        
    # Optionally save results
    with open('action_classifications.json', 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()