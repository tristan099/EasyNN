
## Purpose
The purpose of the files in this directory is to provide
an easy call to the Neural Network constructor to define
what activation functions the hidden and output layers will be made with.

## Functions
```
        Hidden Layer Activation Functions
Tanh - Typical activation function constrains input from -1 to 1
ReLU - Trains faster does not backpropagate negative values
PReLU - parameterized ReLU tries to avoid vanishing gradients
        for deep architecture

        Output Layer Activation functions
Sigmoid - For binary classification. Returns probability from 0 to 1
Softmax - Same as sigmoid but for multi-class classification
Regress - output for regression problems, in the case of continuous target variables.
```

## Optimizers
In addition, each activation function has a series of optimizers that can make
the training process faster.

```
Momentum - Introduces the concept of momentum to the navigation of the loss function,
for smoother gradient descent

Nesterov Momentum - Type of momentum where we look into future to evaluate next training step, 
basically a faster form of momentum

Adagrad, RMSProp, ADAM - these are all general optimizers that, with the exception of ADAM, can be used
in combination with either momentum or Nesterov momentum.
```

## Usage
```
Example for Neural Network
from EasyNN import *

Define your NN architecture 

model=EasyNN(layers=[prelu(indims=2,outdims=40),prelu(indims=40,outdims=40),sigmoid(40,1)],costfunction=BCEC)

train
model.rmsnesttrain(x=X,y=Y,LR=5e-8,epochs=100,mu=.9,ep=1e-9,gam=0.9,taskclassification=True,l1=0,l2=0, dropout=False, p=0)

predict
model.predict(x=X, y=Y, taskclassification=true,costfunction=BCEC)
```