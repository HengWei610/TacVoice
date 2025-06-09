# models/dgin/encoder.py
class DGINEncoder:
    def __init__(self, config):
        self.config = config
        # Initialize layers: GNN, TCN, Attention

    def forward(self, player_states, game_context):
        # Encode spatial features via GNN
        # Encode temporal features via TCN
        # Fuse context features
        # Return latent representations
        pass
