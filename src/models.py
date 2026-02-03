import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 1. The Weighted Average Layer (Specific for WavLM) ---
class WeightedAverageLayer(nn.Module):
    def __init__(self, num_layers=13):
        super().__init__()
        # Learnable weights for each of the 13 layers
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x):
        # x shape: (Batch, 13, 768)
        
        # Softmax ensures weights sum to 1 (interpretable attention)
        w = F.softmax(self.weights, dim=0)
        
        # Broadcast weights: (1, 13, 1) to match (Batch, 13, 768)
        x_weighted = x * w.view(1, -1, 1)
        
        # Sum across layers -> (Batch, 768)
        return torch.sum(x_weighted, dim=1)

# --- 2. The Generic MLP (The Classifier) ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        
        # A standard architecture for medical data (prevent overfitting)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # Helps convergence
            nn.ReLU(),
            nn.Dropout(dropout_rate),   # Critical for small data
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Binary Output (1 logit)
            nn.Linear(hidden_dim // 2, 1) 
        )

    def forward(self, x):
        # x shape: (Batch, Input_Dim)
        return self.net(x)

# --- 3. The WavLM Wrapper ---
class WavLMClassifier(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        # 1. Collapse 13 layers -> 1 layer
        self.aggregator = WeightedAverageLayer(num_layers=13)
        
        # 2. Classify the resulting 768 vector
        self.classifier = SimpleMLP(
            input_dim=768, 
            hidden_dim=hidden_dim, 
            dropout_rate=dropout_rate
        )

    def forward(self, x):
        # x shape: (Batch, 13, 768)
        
        # Aggregate -> (Batch, 768)
        pooled_embedding = self.aggregator(x)
        
        # Classify -> (Batch, 1)
        logits = self.classifier(pooled_embedding)
        
        return logits

# --- 4. Factory Function (Easy to use in training loop) ---
def get_model(model_type, input_dim=None, hidden_dim=128, dropout=0.3):
    """
    Returns the correct model based on the string identifier.
    """
    if model_type == 'wavlm':
        return WavLMClassifier(hidden_dim=hidden_dim, dropout_rate=dropout)
    
    elif model_type in ['roberta', 'classic']:
        if input_dim is None:
            raise ValueError("input_dim must be provided for classic/roberta models")
        return SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
    
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

# --- SANITY CHECK BLOCK ---
if __name__ == "__main__":
    print("--- Model Sanity Checks ---")
    
    # 1. Test WavLM Model
    batch_size = 8
    wavlm_input = torch.randn(batch_size, 13, 768)
    model_wav = get_model('wavlm')
    out_wav = model_wav(wavlm_input)
    print(f"WavLM Input: {wavlm_input.shape} -> Output: {out_wav.shape}")
    assert out_wav.shape == (batch_size, 1)
    
    # 2. Test RoBERTa Model
    roberta_input = torch.randn(batch_size, 768)
    model_rob = get_model('roberta', input_dim=768)
    out_rob = model_rob(roberta_input)
    print(f"RoBERTa Input: {roberta_input.shape} -> Output: {out_rob.shape}")
    assert out_rob.shape == (batch_size, 1)

    # 3. Test Classic Model (e.g., 2072 features)
    classic_input = torch.randn(batch_size, 2072)
    model_cls = get_model('classic', input_dim=2072)
    out_cls = model_cls(classic_input)
    print(f"Classic Input: {classic_input.shape} -> Output: {out_cls.shape}")
    assert out_cls.shape == (batch_size, 1)
    
    print("\n✅ All Model Checks Passed.")