import numpy as np
import matplotlib.pyplot as plt 

# SAMPLE input data
inputs = np.array([[0, 1, 0],
                   [0, 1, 1],
                   [0, 0, 0],
                   [1, 0, 0],
                   [1, 1, 1],
                   [1, 0, 1]])
# SAMPLE output data
outputs = np.array([[0], [0], [0], [1], [1], [1]])


class NeuralNetwork:
    def __init__(self, inputs, outputs):
        self.inputs  = inputs
        self.outputs = outputs
        # initialize weights as .50 for simplicity
        self.weights = np.array([[.50], [.50], [.50]])
        #not using biases for simplicity....also they are not needed
        self.error_history = []
        self.epoch_list = []


    def sigmoid(self, x, deriv=False):
        if deriv == True:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    #Data analyzing
    def feed_forward(self):
        self.hidden = self.sigmoid(np.dot(self.inputs, self.weights))

    # Backpropagation
    def backpropagation(self):
        self.error  = self.outputs - self.hidden
        delta = self.error * self.sigmoid(self.hidden, deriv=True)
        self.weights += np.dot(self.inputs.T, delta)

    # Training on basis of backpropagation and reiterating that process
    def train(self, epochs=25000):
        for epoch in range(epochs):
            self.feed_forward()
            self.backpropagation()    
            
            # keep track of the error history over each epoch for graph plotting
            self.error_history.append(np.average(np.abs(self.error)))
            self.epoch_list.append(epoch)


    # function to predict output on new data    
    def predict(self, new_input):
        prediction = self.sigmoid(np.dot(new_input, self.weights))
        return prediction

# create and train neural network   
NN = NeuralNetwork(inputs, outputs)
NN.train()

#Test examples                              
example = np.array([[1, 1, 0]])     #output should be 1
example_2 = np.array([[0, 1, 1]])   #output should be 0

# print the predictions for both examples  and the expected values  
print(NN.predict(example), ' - Correct: ', example[0][0])
print(NN.predict(example_2), ' - Correct: ', example_2[0][0])

# plot the errors against the no of iterations
plt.figure(figsize=(15,5))
plt.plot(NN.epoch_list, NN.error_history)
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.show()