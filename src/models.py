import torch
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
        A Multi-Layer Perceptron for embedding features.

        Args:
            input_size (int): The number of input features (e.g., num_layers * 768).
            hidden_size (int): The number of neurons in each hidden layer.
            num_hidden_layers (int): The number of hidden layers.
            dropout_rate (float): The dropout rate for regularization.
        """
        super(EmbeddingMLP, self).__init__()

        layers = []

        # Input Layer (Can be larger for higher-dimensional embeddings)
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

class FusionANN(nn.Module):
    def __init__(self, clinical_input_size, embedding_input_size,
                 conv_out_channels=16, conv_kernel_size=3, linear_hidden_size=256, dropout_rate=0.5):
        """
        Fusion model inspired by the paper, using a 1D CNN.
        Fuses clinical and embedding features by treating them as channels.

        Args:
            clinical_input_size (int): The original size of the clinical feature vector.
            embedding_input_size (int): The original size of the embedding feature vector.
            conv_out_channels (int): Number of output channels for the convolutional layer.
            conv_kernel_size (int): Kernel size for the convolutional layer.
            linear_hidden_size (int): Size of the dense layer after convolution.
            dropout_rate (float): Dropout rate for regularization.
        """
        super(FusionANN, self).__init__()
        
        self.clinical_input_size = clinical_input_size
        self.embedding_input_size = embedding_input_size
        
        # The final padded length will be the max of the two inputs
        self.padded_length = max(clinical_input_size, embedding_input_size)
        
        # 1D Convolutional Layer
        # in_channels=2 because we are fusing two feature sets (clinical and embedding)
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=conv_out_channels, 
                               kernel_size=conv_kernel_size, padding='same')
        
        # Adaptive Pooling: This will pool the output of the conv layer to a fixed size,
        # which is robust to changes in padding or kernel size.
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification Head
        self.fc_head = nn.Sequential(
            nn.Linear(conv_out_channels, linear_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(linear_hidden_size, 1) # 1 output for binary classification
        )

    def forward(self, x_clinical, x_embedding):
        # 1. Pad the shorter feature vector to match the length of the longer one.
        # We assume embedding features are longer.
        if self.clinical_input_size < self.padded_length:
            # F.pad format: (pad_left, pad_right)
            pad_amount = self.padded_length - self.clinical_input_size
            x_clinical = F.pad(x_clinical, (0, pad_amount), "constant", 0)

        # 2. Reshape and stack to create the (batch, channels, length) tensor
        # Add a channel dimension to each
        x_clinical = x_clinical.unsqueeze(1)    # Shape: (batch_size, 1, padded_length)
        x_embedding = x_embedding.unsqueeze(1)  # Shape: (batch_size, 1, padded_length)
        
        # Stack along the channel dimension
        x_fused = torch.cat((x_clinical, x_embedding), dim=1) # Shape: (batch_size, 2, padded_length)
        
        # 3. Pass through the network
        x = self.conv1(x_fused)
        x = F.relu(x)
        x = self.pool(x) # Shape: (batch_size, conv_out_channels, 1)
        
        # Flatten the output from the pooling layer before the linear head
        x = x.squeeze(-1) # Shape: (batch_size, conv_out_channels)
        
        # 4. Final classification
        output = self.fc_head(x)
        
        return output

class WeightedAverage(torch.nn.Module):
    def __init__(self, num_layers=13):
        super().__init__()
        # Create a learnable parameter vector, one weight per layer
        self.weights = torch.nn.Parameter(data=torch.ones((num_layers,)))

    def forward(self, x):
        # x has shape: (batch_size, num_layers, features) e.g. (32, 13, 768)
        
        # Turn weights into a probability distribution that sums to 1
        w = torch.nn.functional.softmax(self.weights, dim=0)
        
        # Multiply each layer's features by its learned weight.
        # w[None, :, None] broadcasts the weights to shape (1, num_layers, 1)
        x_weighted = x * w[None, :, None]
        
        # Sum across the layer dimension to get the final weighted average
        # The output has shape (batch_size, features) e.g. (32, 768)
        return torch.sum(x_weighted, dim=1)

# --- This is our new main model ---
class WeightedEmbeddingMLP(nn.Module):
    def __init__(self, features_per_layer, hidden_size, num_layers, dropout_rate):
        super().__init__()

        # Instantiate the weighted average layer
        self.weighted_average = WeightedAverage(num_layers)

        # The MLP body now takes the output of the weighted average layer as input
        # Its input dimension is features_per_layer (e.g., 768), NOT the flattened size.
        self.network_body = nn.Sequential(
            nn.Linear(features_per_layer, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1) # Final output layer
        )

    def forward(self, x):
        # 1. Apply the learned weighted average to the input layers
        weighted_x = self.weighted_average(x)
        
        # 2. Pass the result through the standard MLP
        output = self.network_body(weighted_x)
        
        return output