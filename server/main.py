from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
app = Flask(__name__)
cors = CORS(app,origins='*')

import torch
import torch.nn as nn

@app.route("/api/number",methods=['GET'])
def number():
    return jsonify(
        {
            "number":[
                predicted_digit
            ]
        }
    )


# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model (should match the architecture used for training)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = SimpleCNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()

# Function to test custom input
def test_custom_input(model, custom_array):
    # Preprocess the custom array
    custom_tensor = torch.tensor(custom_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
    custom_tensor = custom_tensor.to(device)  # Move to GPU if available

    # Make predictions
    with torch.no_grad():
        output = model(custom_tensor)
        _, predicted = torch.max(output.data, 1)
    
    return predicted.item()

predicted_digit = 0

@app.route('/save-drawing', methods=['POST'])
def save_drawing():
    data = request.get_json()  # Get the JSON data from the request
    #print(data)  # You can process the data as needed (e.g., save it to a file or database)
    predicted_digit = test_custom_input(model, data)
    print(f'Predicted digit: {predicted_digit}')
    # Send a response back to the client
    return jsonify({'message': 'Drawing saved successfully!', 'result_number': predicted_digit}), 200

if __name__ == "__main__":
    app.run(debug=True, port= 8080)






