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
    

class LearnableStatPoolingMLP(nn.Module):
    def __init__(self, num_layers, features_per_layer, hidden_size, dropout_rate):
        super().__init__()
        
        # Part 1: The learnable layer averager
        self.layer_averager = WeightedAverage(num_layers)
        
        # Part 2: The final MLP classifier head
        # The input size is features_per_layer * 2 (for mean + std dev)
        mlp_input_size = features_per_layer * 2
        
        self.classifier = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, x, lengths):
        # x shape: (batch_size, seq_len, num_layers, features)
        
        # 1. Apply learnable weighted average across the layer dimension
        batch_size, seq_len, num_layers, features = x.shape
        x_reshaped = x.view(batch_size * seq_len, num_layers, features)
        word_vectors_reshaped = self.layer_averager(x_reshaped)
        word_vectors = word_vectors_reshaped.view(batch_size, seq_len, features)
        
        # 2. Perform Statistical Pooling (mean + std), ignoring padding
        pooled_vectors = []
        for i in range(batch_size):
            # Get the non-padded sequence for this sample
            sequence = word_vectors[i, :lengths[i], :]
            
            # The number of actual words in this sequence
            num_words = sequence.shape[0]

            # This should not happen with a good dataloader, but it's a robust safety check
            if num_words == 0:
                # If there are no words, the feature is just zeros
                # The size is features * 2 (for mean + std)
                final_features = torch.zeros(features * 2, device=x.device, dtype=x.dtype)
            
            else:
                # Calculate the mean vector
                mean_vec = torch.mean(sequence, dim=0)
                
                # --- THIS IS THE FIX ---
                if num_words > 1:
                    # If we have more than one word, calculate std dev normally
                    std_vec = torch.std(sequence, dim=0)
                else:
                    # If there is only one word, its standard deviation is a vector of zeros.
                    # We create a zero tensor with the same shape, device, and dtype as the mean vector.
                    std_vec = torch.zeros_like(mean_vec)
                # --- END FIX ---
                
                # Concatenate the final feature vector
                final_features = torch.cat([mean_vec, std_vec])
            
            pooled_vectors.append(final_features)
            
        # Stack the results into a single batch tensor
        # -> (batch_size, features * 2)
        final_batch_features = torch.stack(pooled_vectors)
        
        # 3. Classify the final feature vector
        return self.classifier(final_batch_features)


class FusionMLP(nn.Module):
    def __init__(self, acoustic_dim, linguistic_dim, hidden_size, dropout_rate):
        super().__init__()
        
        # Branch 1: Acoustic Expert
        self.acoustic_branch = nn.Sequential(
            nn.Linear(acoustic_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Branch 2: Linguistic Expert
        self.linguistic_branch = nn.Sequential(
            nn.Linear(linguistic_dim, hidden_size // 2), # Smaller branch for smaller input
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Head: Fuses the outputs of the two branches
        fusion_input_dim = hidden_size + (hidden_size // 2)
        self.fusion_head = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1) # Final output
        )
        
    def forward(self, x_acoustic, x_linguistic):
        # Process each modality independently
        acoustic_out = self.acoustic_branch(x_acoustic)
        linguistic_out = self.linguistic_branch(x_linguistic)
        
        # Concatenate (fuse) the expert opinions
        fused_vector = torch.cat([acoustic_out, linguistic_out], dim=1)
        
        # Make the final decision
        output = self.fusion_head(fused_vector)
        return output