import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np # <--- 1. ADDED THIS

# --- 1. The Weighted Average Layer ---
class WeightedAverageLayer(nn.Module):
    def __init__(self, num_layers=13):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, x):
        # x shape: (Batch, 13, 768)
        w = F.softmax(self.weights, dim=0)
        x_weighted = x * w.view(1, -1, 1)
        return torch.sum(x_weighted, dim=1)

# --- 2. The Generic MLP ---
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 3. The WavLM Wrapper ---
class WavLMClassifier(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.aggregator = WeightedAverageLayer(num_layers=13)
        self.classifier = SimpleMLP(768, hidden_dim, dropout_rate)

    def forward(self, x):
        pooled_embedding = self.aggregator(x)
        return self.classifier(pooled_embedding)
    
    # ---------------------------------------------------------
    # 2. NEW METHOD: Allows us to save weights to CSV
    # ---------------------------------------------------------
    def get_layer_weights(self):
        """Returns the 13 softmaxed weights as a numpy array."""
        w = F.softmax(self.aggregator.weights, dim=0)
        return w.detach().cpu().numpy()

# --- 4. Gated Multimodal Unit (Fusion) ---
class GMU(nn.Module):
    def __init__(self, dim_audio, dim_text, hidden_dim, dropout=0.2):
        super().__init__()
        self.linear_audio = nn.Linear(dim_audio, hidden_dim)
        self.linear_text = nn.Linear(dim_text, hidden_dim)
        
        # Stability: LayerNorm helps when modalities have different scales
        self.ln_audio = nn.LayerNorm(hidden_dim)
        self.ln_text = nn.LayerNorm(hidden_dim)
        
        self.z_gate = nn.Linear(dim_audio + dim_text, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Store the last gate value for logging
        self.last_z_mean = 0.0

    def forward(self, audio, text):
        # 1. Project and Normalize
        h_audio = torch.tanh(self.ln_audio(self.linear_audio(audio)))
        h_text = torch.tanh(self.ln_text(self.linear_text(text)))
        
        # 2. Calculate Gate
        combined = torch.cat([audio, text], dim=1)
        z = torch.sigmoid(self.z_gate(combined))
        
        # Save mean gate value for interpretability (how much it trusts audio)
        self.last_z_mean = z.mean().item()
        
        # 3. Dynamic Fusion
        h = z * h_audio + (1 - z) * h_text
        return self.dropout(h)

class FusionClassifier(nn.Module):
    def __init__(self, hidden_dim=128, dropout_rate=0.3):
        super().__init__()
        self.wavlm_agg = WeightedAverageLayer(num_layers=13)
        self.gmu = GMU(dim_audio=768, dim_text=768, hidden_dim=hidden_dim, dropout=dropout_rate)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x_wavlm, x_roberta):
        audio_emb = self.wavlm_agg(x_wavlm)
        fused_vector = self.gmu(audio_emb, x_roberta)
        return self.classifier(fused_vector)

    def get_layer_weights(self):
        # Returns WavLM weights (existing logic)
        return F.softmax(self.wavlm_agg.weights, dim=0).detach().cpu().numpy()

    def get_gate_value(self):
        # Returns the mean trust in Audio (0 to 1)
        return self.gmu.last_z_mean

# --- Factory Function ---
def get_model(model_type, input_dim=None, hidden_dim=128, dropout=0.3):
    if model_type == 'wavlm':
        return WavLMClassifier(hidden_dim=hidden_dim, dropout_rate=dropout)
    elif model_type == 'fusion':
        return FusionClassifier(hidden_dim=hidden_dim, dropout_rate=dropout)
    elif model_type in ['roberta', 'classic']:
        return SimpleMLP(input_dim=input_dim, hidden_dim=hidden_dim, dropout_rate=dropout)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")