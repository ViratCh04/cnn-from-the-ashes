import numpy as np

class Conv3x3:
    """Creates a convolution layer using 3x3 filters"""
    def __init__(self, numFilters):
        self.numFilters = numFilters
        # filters is a 3D array with (numFilters, 3, 3) dims
        self.filters = np.random.randn(numFilters, 3, 3) / 9
        # Dividing by 9 to reduce variance of initial random values(Refer to Glorot Initialisation for more)
        
    def iterateRegions(self, image):
        """
        Generates all possible 3x3 image regions using valid padding(no padding)
        - image is a 2D np array
        """
        
        h, w = image.shape
        for i in range(h - 2):
            for j in range(w - 2):
                imRegion = image[i:(i + 3), j:(j + 3)]
                yield imRegion, i, j
                
    def forward(self, input):
        """
        Performs a forward pass of the conv layer with 8 filters using the given input.
        That is, the filters convolve with the generated regions of the input(from iterateRegions())
        Returns a 3D np array with dimensions (h, w, numFilters)(26, 26, 8 for MNIST images)
        - input is a 2D np array
        """
        self.lastInput = input
        
        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.numFilters))
        
        for imageRegion, i , j in self.iterateRegions(input):
            output[i, j] = np.sum(imageRegion * self.filters, axis=(1, 2))
        
        return output
    
    def backprop(self, dL_dout, learnRate):
        """
        Performs a backward pass of the conv layer.
        - dL_dout is the loss gradient for this layer's outputs which this fn receives as input.
        - learnRate is a float.
        """
        dL_dfilters = np.zeros(self.filters.shape)
        
        for imageRegion, i, j in self.iterateRegions(self.lastInput):
            for f in range(self.numFilters):
                dL_dfilters[f] += dL_dout[i, j, f] * imageRegion
                
        # Update filters
        self.filters -= learnRate * dL_dfilters
        
        # No need whatsoever to return anything(loss gradients) as this layer is the very first layer in the CNN
        return None