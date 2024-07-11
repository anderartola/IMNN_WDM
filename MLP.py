import numpy as np
import torch
import torch.nn as nn
import numpy as np




class MLP(nn.Module):
    
    """
    A neural network with multiple layers and activation functions.

    Args:
    data_dim (int): The dimensionality of the input data.

    Attributes:
    fc1 (nn.Linear): The first fully connected layer with data_dim input nodes and 512 output nodes.
    dropout1 (nn.Dropout): A dropout layer with a probability of 0.4 for dropping out nodes.
    prelu1 (nn.PReLU): A Parametric ReLU activation function for the first layer.
    fc2 (nn.Linear): The second fully connected layer with 512 input nodes and 512 output nodes.
    dropout2 (nn.Dropout): A dropout layer with a probability of 0.4 for dropping out nodes.
    prelu2 (nn.PReLU): A Parametric ReLU activation function for the second layer.
    fc3 (nn.Linear): The third fully connected layer with 512 input nodes and 256 output nodes.
    dropout3 (nn.Dropout): A dropout layer with a probability of 0.4 for dropping out nodes.
    prelu3 (nn.PReLU): A Parametric ReLU activation function for the third layer.
    fc4 (nn.Linear): The fourth fully connected layer with 256 input nodes and 1 output node.
    leaky_relu (nn.ELU): A Leaky ReLU activation function.

    Methods:
    forward(self, x): Forward pass of the neural network. Takes input x, applies the layers in sequence, and returns the output.
    """

    def __init__(self, data_dim):
        
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(data_dim, 512)
        self.dropout1 = nn.Dropout(p=0.4)
        self.prelu1=nn.PReLU()
    
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(p=0.4)  
        self.prelu2=nn.PReLU()
        
        self.fc3 = nn.Linear(512, 256)
        self.dropout3 = nn.Dropout(p=0.4)       
        self.prelu3=nn.PReLU()
        
        self.fc4 = nn.Linear(256, 1)

        self.leaky_relu = nn.ELU()  


    def forward(self, x):
        x = self.prelu1(self.fc1(x))
        x=self.dropout1(x)
        
        x = self.prelu2(self.fc2(x))
        x=self.dropout2(x)
        
        x = self.prelu3(self.fc3(x))
        x=self.dropout3(x)  
        
        x=self.fc4(x)

        return x