import numpy as np

class MaxPool2:
    # A Max Pooling layer using a pool size of 2(divides the input by 2)
    def iterateRegions(self, image):
        """
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2D np array
        """
        h, w, _ = image.shape
        newH = h // 2
        newW = w // 2
        
        for i in range(newH):
            for j in range(newW):
                imRegion = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield imRegion, i, j
                
    def forward(self, input):
        """
        Performs a forward pass of maxpool layer using given input.
        Returns a 3D np array with dims (h/2, w/2, numFilters).
        - input is a 3D np array with dims (h, w, numFilters)
        """
        self.lastInput = input
        
        h, w, numFilters = input.shape
        output = np.zeros((h // 2, w // 2, numFilters))
        for imRegion, i, j in self.iterateRegions(input):
            output[i, j] = np.amax(imRegion, axis=(0, 1))
            
        return output
    
    def backprop(self, dL_dout):
        """
        Performs a backward pass of maxpool layer.
        Returns the loss gradient for this layer's inputs after receiving this layer's outputs.
        - dL_dout is the loss gradient for this layer's outputs
        """
        dL_dinput = np.zeros(self.lastInput.shape)
        
        for imRegion, i, j in self.iterateRegions(self.lastInput):
            h, w, f = imRegion.shape
            amax = np.amax(imRegion, axis=(0, 1))
            
            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        if imRegion[i2, j2, f2] == amax[f2]:
                        # Consult to p2 of referenced article for more in depth explanation for how this step came to be
                            dL_dinput[i * 2 + i2, j * 2 + j2, f2] = dL_dout[i, j, f2]
        
        return dL_dinput