# train_action_classifier.py
import numpy as np
import tensorflow as tf
from action_classifier import (
    load_and_preprocess_data, 
    create_model,
    extract_features
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def train_classifier():
    # Load and preprocess data
    sequences, labels = load_and_preprocess_data('30fps1920.json')
    
    # Convert string labels to integers
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Save label mapping
    label_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    np.save('label_mapping.npy', label_mapping)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        sequences, labels_encoded, test_size=0.2, random_state=42
    )
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])  # (sequence_length, num_features)
    num_classes = len(label_encoder.classes_)
    model = create_model(input_shape, num_classes)
    
    # Add data augmentation
    def augment_sequence(sequence):
        # Add random noise to features
        noise = np.random.normal(0, 0.01, sequence.shape)
        return sequence + noise
    
    # Augment training data
    X_train_augmented = []
    y_train_augmented = []
    
    for seq, label in zip(X_train, y_train):
        # Original sequence
        X_train_augmented.append(seq)
        y_train_augmented.append(label)
        
        # Augmented sequences
        for _ in range(2):  # Create 2 augmented versions
            X_train_augmented.append(augment_sequence(seq))
            y_train_augmented.append(label)
    
    X_train_augmented = np.array(X_train_augmented)
    y_train_augmented = np.array(y_train_augmented)
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True
        )
    ]
    
    # Train model
    history = model.fit(
        X_train_augmented, 
        y_train_augmented,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'\nTest accuracy: {test_acc:.4f}')
    
    # Save model
    model.save('squash_action_classifier.h5')
    
    return history, label_mapping

if __name__ == '__main__':
    history, label_mapping = train_classifier()
    print("\nLabel mapping:", label_mapping)