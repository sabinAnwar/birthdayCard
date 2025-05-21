import numpy as np

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, X):
        self.hidden = self.sigmoid(np.dot(X, self.weights1))
        output = self.sigmoid(np.dot(self.hidden, self.weights2))
        return output

# Create a simple dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data
y = np.array([[0], [1], [1], [0]])  # XOR output

# Create and train the network
nn = SimpleNeuralNetwork(2, 4, 1)

# Make a prediction
prediction = nn.forward(X[0])
print(f"Input: {X[0]}")
print(f"Prediction: {prediction[0]:.4f}") 