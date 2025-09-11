import torch.nn as nn
import torch.nn.functional as F

class ExplainableMLP(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_hidden_layers=2, dropout_rate=0.5):
        """
        A simple Multi-Layer Perceptron for binary classification.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of neurons in each hidden layer.
            num_hidden_layers (int): The number of hidden layers.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(ExplainableMLP, self).__init__()
        
        layers = []
        
        # Input Layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden Layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output Layer: 1 neuron for binary classification
        layers.append(nn.Linear(hidden_size, 1))
        
        # Combine all layers into a sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward pass of the model.
        The output is a raw logit. The loss function will handle the sigmoid activation.
        """
        return self.network(x)

class EmbeddingMLP(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_hidden_layers=3, dropout_rate=0.5):
        """
        A Multi-Layer Perceptron for embedding features (typically higher dimensional).

        Args:
            input_size (int): The number of input features (e.g., num_layers * 768).
            hidden_size (int): The number of neurons in each hidden layer.
            num_hidden_layers (int): The number of hidden layers.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(EmbeddingMLP, self).__init__()
        
        layers = []
        
        # Input Layer (Can be larger for high-dimensional embeddings)
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        # Hidden Layers
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            
        # Output Layer: 1 neuron for binary classification
        layers.append(nn.Linear(hidden_size, 1))
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        The forward pass of the model.
        The output is a raw logit.
        """
        return self.network(x)
