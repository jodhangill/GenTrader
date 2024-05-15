import backtrader as bt
import numpy as np

class NeuralNetwork(bt.Strategy):
    params = (
        # Weights from input node 1 to hidden nodes
        ('w_i1_h1', 0.022940338148537076),
        ('w_i1_h2', -0.07035362318571622),
        ('w_i1_h3', -0.05301731144816083),
        ('w_i1_h4', -0.4404651260276612),
        ('w_i1_h5', -0.1357041147928825),
        ('w_i1_h6', 0.3823854704227519),
        ('w_i1_h7', 0.04838633304274786),
        ('w_i1_h8', -0.18367352065254944),
    
        # Weights from input node 2 to hidden nodes
        ('w_i2_h1', -0.01319879950625087),
        ('w_i2_h2', 0.0373477384605171),
        ('w_i2_h3', -0.06372517623637722),
        ('w_i2_h4', 1.229058769628469),
        ('w_i2_h5', -0.01567991823866113),
        ('w_i2_h6', 0.18671636140895606),
        ('w_i2_h7', 0.20314328646037375),
        ('w_i2_h8', -0.007511519507356345),
    
        # Weights from input node 3 to hidden nodes
        ('w_i3_h1', 0.013563586691604988),
        ('w_i3_h2', 0.021835282695328324),
        ('w_i3_h3', -0.008156929514610141),
        ('w_i3_h4', 0.34986384139905397),
        ('w_i3_h5', 0.04835900302225245),
        ('w_i3_h6', 0.2694644287528727),
        ('w_i3_h7', 0.34249157408931685),
        ('w_i3_h8', -0.06579828703138527),
    
        # Weights from input node 4 to hidden nodes
        ('w_i4_h1', -0.2564035695012371),
        ('w_i4_h2', -0.1342525855970344),
        ('w_i4_h3', -0.7186776623643717),
        ('w_i4_h4', -0.36109499687154956),
        ('w_i4_h5', 0.10320856327749003),
        ('w_i4_h6', -0.4286492765404204),
        ('w_i4_h7', 0.019891993003599853),
        ('w_i4_h8', -0.3841094959261574),
    
        # Weights from input node 5 to hidden nodes
        ('w_i5_h1', -0.03338862121523939),
        ('w_i5_h2', 0.05021675330878315),
        ('w_i5_h3', -0.10494658301325863),
        ('w_i5_h4', 0.06590230614226508),
        ('w_i5_h5', -0.16825499947171205),
        ('w_i5_h6', -0.07077614306361547),
        ('w_i5_h7', 0.018646747799504634),
        ('w_i5_h8', -0.06518186325247964),
    
        # Weights from hidden nodes to output node
        ('w_h1_o1', -0.0009385657505354857),
        ('w_h2_o1', -0.08171781067309344),
        ('w_h3_o1', 0.09336333546543321),
        ('w_h4_o1', -0.5070232276639965),
        ('w_h5_o1', 0.37483051887805796),
        ('w_h6_o1', 0.11409691248143092),
        ('w_h7_o1', -0.26516713993890434),
        ('w_h8_o1', 0.23847312698953682)
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
        
        if outputs > 0.5:
            self.buy()
        if outputs < 0.5:
            self.sell()
