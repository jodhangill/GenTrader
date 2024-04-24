import backtrader as bt
import numpy as np

class NeuralNetwork(bt.Strategy):

    def __init__(self):
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=15)
        self.atr = bt.indicators.ATR(self.datas[0])
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.datas[0])
        self.rsi = bt.indicators.RSI(self.datas[0])
        self.volume = self.datas[0].volume

        # Initialize variables to store means and standard deviations
        self.means = np.zeros(5)
        self.stds = np.ones(5) 
        
        # Randomize weights with 5 inputs, 1 output
        self.weights_hidden = np.random.randn(5, 1)

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def think(self, inputs):
        """Compute the output of the neural network."""
        # Convert inputs to numpy array for easier computation
        inputs = np.array(inputs)

        # Compute activations of neurons in the output layer
        output_activations = self.sigmoid(np.dot(inputs, self.weights_hidden))
        
        return output_activations

    def scale_data(self, data):

        # Concatenate input features into a single array
        inputs = np.array(data)

        # Scale input features using stored means and standard deviations
        scaled_inputs = (inputs - self.means) / self.stds

        return scaled_inputs

    def update_statistics(self, data):
        # Update means
        self.means = np.array([np.mean([x, self.means[i]]) for i, x in enumerate(data)])

        # Update standard deviations
        self.stds = np.array([np.std([x, self.stds[i]]) for i, x in enumerate(data)])

    def next(self):
        # Get current values
        inputs = [
            self.sma[0],
            self.atr[0],
            self.adx[0],
            self.rsi[0],
            self.volume[0]
        ]

        self.update_statistics(inputs)
        scaled_inputs = self.scale_data(inputs)

        outputs = self.think(scaled_inputs)
        
        if outputs > 0.5:
            self.buy()
        if outputs < 0.5:
            self.sell()