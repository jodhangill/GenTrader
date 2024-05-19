import backtrader as bt
import numpy as np

class NeuralNetwork(bt.Strategy):
    '''
    Neural Network Strategy:
    - Uses a simple neural network with one hidden layer to make trading decisions.
    - The network has 5 input nodes, 8 hidden nodes, and 1 output node.
    - Buys when the output of the neural network is greater than 0.5 and sells when it's less than 0.5.

    Parameters:
    - w_i1_h1, ..., w_i5_h8: Weights from input nodes to hidden nodes (default: 0.0).
    - w_h1_o1, ..., w_h8_o1: Weights from hidden nodes to output node (default: 0.0).
    '''
    params = (
        # Weights from input node 1 to hidden nodes
        ('w_i1_h1', 0.0),
        ('w_i1_h2', 0.0),
        ('w_i1_h3', 0.0),
        ('w_i1_h4', 0.0),
        ('w_i1_h5', 0.0),
        ('w_i1_h6', 0.0),
        ('w_i1_h7', 0.0),
        ('w_i1_h8', 0.0),
    
        # Weights from input node 2 to hidden nodes
        ('w_i2_h1', 0.0),
        ('w_i2_h2', 0.0),
        ('w_i2_h3', 0.0),
        ('w_i2_h4', 0.0),
        ('w_i2_h5', 0.0),
        ('w_i2_h6', 0.0),
        ('w_i2_h7', 0.0),
        ('w_i2_h8', 0.0),
    
        # Weights from input node 3 to hidden nodes
        ('w_i3_h1', 0.0),
        ('w_i3_h2', 0.0),
        ('w_i3_h3', 0.0),
        ('w_i3_h4', 0.0),
        ('w_i3_h5', 0.0),
        ('w_i3_h6', 0.0),
        ('w_i3_h7', 0.0),
        ('w_i3_h8', 0.0),
    
        # Weights from input node 4 to hidden nodes
        ('w_i4_h1', 0.0),
        ('w_i4_h2', 0.0),
        ('w_i4_h3', 0.0),
        ('w_i4_h4', 0.0),
        ('w_i4_h5', 0.0),
        ('w_i4_h6', 0.0),
        ('w_i4_h7', 0.0),
        ('w_i4_h8', 0.0),
    
        # Weights from input node 5 to hidden nodes
        ('w_i5_h1', 0.0),
        ('w_i5_h2', 0.0),
        ('w_i5_h3', 0.0),
        ('w_i5_h4', 0.0),
        ('w_i5_h5', 0.0),
        ('w_i5_h6', 0.0),
        ('w_i5_h7', 0.0),
        ('w_i5_h8', 0.0),
    
        # Weights from hidden nodes to output node
        ('w_h1_o1', 0.0),
        ('w_h2_o1', 0.0),
        ('w_h3_o1', 0.0),
        ('w_h4_o1', 0.0),
        ('w_h5_o1', 0.0),
        ('w_h6_o1', 0.0),
        ('w_h7_o1', 0.0),
        ('w_h8_o1', 0.0)
    )

    def __init__(self):
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=15)
        self.atr = bt.indicators.ATR(self.datas[0])
        self.adx = bt.indicators.AverageDirectionalMovementIndex(self.datas[0])
        self.rsi = bt.indicators.RSI(self.datas[0])
        self.volume = self.datas[0].volume

        # Initialize variables to store means and standard deviations
        self.means = np.zeros(5)
        self.stds = np.ones(5) 
        
        # Set weights with 5 inputs, 8 hidden nodes
        self.weights_hidden = np.array([
            [self.p.w_i1_h1, self.p.w_i1_h2, self.p.w_i1_h3, self.p.w_i1_h4, self.p.w_i1_h5, self.p.w_i1_h6, self.p.w_i1_h7, self.p.w_i1_h8],
            [self.p.w_i2_h1, self.p.w_i2_h2, self.p.w_i2_h3, self.p.w_i2_h4, self.p.w_i2_h5, self.p.w_i2_h6, self.p.w_i2_h7, self.p.w_i2_h8],
            [self.p.w_i3_h1, self.p.w_i3_h2, self.p.w_i3_h3, self.p.w_i3_h4, self.p.w_i3_h5, self.p.w_i3_h6, self.p.w_i3_h7, self.p.w_i3_h8],
            [self.p.w_i4_h1, self.p.w_i4_h2, self.p.w_i4_h3, self.p.w_i4_h4, self.p.w_i4_h5, self.p.w_i4_h6, self.p.w_i4_h7, self.p.w_i4_h8],
            [self.p.w_i5_h1, self.p.w_i5_h2, self.p.w_i5_h3, self.p.w_i5_h4, self.p.w_i5_h5, self.p.w_i5_h6, self.p.w_i5_h7, self.p.w_i5_h8]
        ])
        # Set weights with 8 hidden nodes, 1 output
        self.weights_output = np.array([
            [self.p.w_h1_o1],
            [self.p.w_h2_o1],
            [self.p.w_h3_o1],
            [self.p.w_h4_o1],
            [self.p.w_h5_o1],
            [self.p.w_h6_o1],
            [self.p.w_h7_o1],
            [self.p.w_h8_o1],
        ])

    def sigmoid(self, x):
        """Sigmoid activation function."""
        # Avoid overflow and underflow
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def think(self, inputs):
        """Compute the output of the neural network."""
        # Convert inputs to numpy array for easier computation
        inputs = np.array(inputs)

        # Compute activations of neurons in the hidden layer
        hidden_activations = self.sigmoid(np.dot(inputs, self.weights_hidden))
        
        # Compute activations of neurons in the output layer
        output_activations = self.sigmoid(np.dot(hidden_activations, self.weights_output))
        
        return output_activations

    def scale_data(self, data):
        """Scale each of the input data"""
        # Concatenate input features into a single array
        inputs = np.array(data)

        # Scale input features using stored means and standard deviations
        scaled_inputs = (inputs - self.means) / self.stds

        return scaled_inputs

    def update_statistics(self, data):
        """Maintain ongoing mean and standard deviation values"""
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
        
        self.close()
        if outputs > 0.5:
            self.buy()
        if outputs < 0.5:
            self.sell()
