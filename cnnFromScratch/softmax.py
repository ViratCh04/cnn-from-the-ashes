import numpy as np

class Softmax:
    """
    A standard fully-connected(dense) layer with softmax activation.
    """
    def __init__(self, inputLen, nodes):
        # Dividing by inputLen to reduce variance of initial random values
        self.weights = np.random.randn(inputLen, nodes) / inputLen
        self.biases = np.zeros(nodes)
        
    def forward(self, input):
        """
        Performs a forward pass of softmax layer using given input.
        Returns a 1D np array containing the respective probability values.
        - input can be any array with any dimensions.
        """
        self.lastInputShape = input.shape
        
        input = input.flatten()
        self.lastInput = input
        
        inputLen, nodes = self.weights.shape
        
        totals = np.dot(input, self.weights) + self.biases
        self.lastTotals = totals
        
        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)
    
    def backprop(self, dL_dout, learnRate):
        """
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs and receives this layer's outputs as input.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learnRate is a float
        """
        
        # We know only 1 element of dL_dout will be nonzero, which is the correct class.
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue
            
            # e^totals
            t_exp = np.exp(self.lastTotals)
            
            # Sum of all e^totals
            S = np.sum(t_exp)
            
            # Gradients of out[i] against totals
            dout_dt = -t_exp[i] * t_exp / (S ** 2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)
            
            # Gradients of totals against weights/biases/input, from eqn Totals = w * input + b
            dt_dw = self.lastInput
            dt_db = 1
            dt_dinputs = self.weights
            
            # Gradients of loss against totals
            dL_dt = gradient * dout_dt
            
            # Gradients of loss against weights/biases/input
            dL_dw = dt_dw[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt
            
            # Update weights/biases
            self.weights -= learnRate * dL_dw
            self.biases -= learnRate * dL_db
            return dL_dinputs.reshape(self.lastInputShape)