# cnn-from-the-ashes
This implementation was done in an attempt to understand how CNNs work at the backend. The main query which sparked this confusion of its operation being so different from Neural Networks was that I did not quite catch how weights were represented in the form filter matrices in memory, unlike weights in MLP-like ANNs which just represent interconnections of neurons among layers.

While the query still remains unsatisfied to an extent, following this implementation step by step helped in making me aware of how many other things are different in CNNs than your vanilla Neural Networks. 

Do not claim to do anything extraordinary with this, though I want to make it run faster and more optimized besides making available a variety of optimizers and other key arguments which can be passed on to this CNN.

Currrently, this thing gives a test accuracy of 76.8 on 3 epochs from the last test when stacked against the same cnn implementation in keras using an ADAM_v2 optimizer and a batch size of 32 which gives an impressive accuracy of 97+% on testing on MNIST dataset.