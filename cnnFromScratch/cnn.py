import mnist
import numpy as np
from conv import Conv3x3
from maxpool import MaxPool2
from softmax import Softmax

# We only use the first 1k examples of each set in the interest of time.
# Feel free to change this if you want.
trainImages = mnist.train_images()[:500]
trainLabels = mnist.train_labels()[:500]
testImages = mnist.test_images()[:500]
testLabels = mnist.test_labels()[:500]

conv = Conv3x3(8)                  # 28x28x1 -> 26x26x8
pool = MaxPool2()                  # 26x26x8 -> 13x13x8
softmax = Softmax(13 * 13 * 8, 10) # 13x13x8 -> 10

def forward(image, label):
    '''
    Completes a forward pass of the CNN and calculates the accuracy and
    cross-entropy loss.
    - image is a 2d numpy array
    - label is a digit
    '''
    # We transform the image from [0, 255] to [-0.5, 0.5] to make it easier
    # to work with. This is standard practice.
    out = conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = softmax.forward(out)

    # Calculate cross-entropy loss and accuracy. np.log() is the natural log.
    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

def train(im, label, lr=0.005):
    '''
    Completes a full training step on the given image and label.
    Returns the cross-entropy loss and accuracy.
    - image is a 2d numpy array
    - label is a digit
    - lr is the learning rate
    '''
    # Forward
    out, loss, acc = forward(im, label)

    # Calculate initial gradient
    gradient = np.zeros(10)
    gradient[label] = -1 / out[label]

    # Backprop
    gradient = softmax.backprop(gradient, lr)
    gradient = pool.backprop(gradient)
    gradient = conv.backprop(gradient, lr)

    return loss, acc

print('CNN initialized!')

# Train the CNN for 3 epochs
for epoch in range(3):
    print('--- Epoch %d ---' % (epoch + 1))

    # Shuffle the training data
    permutation = np.random.permutation(len(trainImages))
    trainImages = trainImages[permutation]
    trainLabels = trainLabels[permutation]

    # Train!
    loss = 0
    numCorrect = 0
    for i, (im, label) in enumerate(zip(trainImages, trainLabels)):
        if i > 0 and i % 100 == 99:
            print("[Step %d] Past 100 steps: Average Loss %.3f  | Accuracy: %d%%" %
            (i + 1, loss / 100, numCorrect)
            )
            loss = 0
            numCorrect = 0

        l, acc = train(im, label)
        loss += l
        numCorrect += acc

# Test the CNN
print('\n--- Testing the CNN ---')
loss = 0
numCorrect = 0
for im, label in zip(testImages, testLabels):
    _, l, acc = forward(im, label)
    loss += l
    numCorrect += acc

numTests = len(testImages)
print('Test Loss:', loss / numTests)
print('Test Accuracy:', numCorrect / numTests)