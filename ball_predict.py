import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.readlines()
    # Convert the data to a list of integers
    data = [int(line.strip()) for line in data]
    # Group the data into pairs of coordinates
    positions = [(data[i], data[i+1]) for i in range(0, len(data), 2)]
    return positions

def prepare_dataset(positions):
    X = []
    y = []
    
    for i in range(len(positions) - 2):
        X.append(positions[i:i+2])
        y.append(positions[i+2])
    
    # Flatten the input pairs
    X = np.array(X).reshape(-1, 4)
    y = np.array(y)
    
    return X, y

def main():
    positions = load_data('output/ball.txt')
    print(positions)
    X, y = prepare_dataset(positions)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict the next positions
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
    
    # Save the trained model to a file
    joblib.dump(model, 'ball_position_predictor.pkl')

if __name__ == "__main__":
    main()