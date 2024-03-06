from torch import nn
from .modules import Attention, FeedForward, PreNorm

# Definition of the Transformer class
class Transformer(nn.Module):
    def __init__(
        self,
        dim, # Dimensionality of input embeddings
        depth, # Number of layers in the transformer
        heads, # Number of attention heads in the multi-head attention mechanism
        mlp_ratio=4.0, # Ratio of feedforward network dimensionality to the dimensionality of input embeddings
        attn_dropout=0.0, # Dropout rate for the attention mechanism
        dropout=0.0, # General dropout rate for the transformer
        qkv_bias=True, # Whether to add a bias to the query, key, value projections in the attention mechanism
        revised=False, # Whether to use a revised version of the feedforward network
    ):
        super().__init__()
        self.layers = nn.ModuleList([]) # Initialize a list to hold the transformer's layers

        self.attn_weights_list = [] # Initialize an empty list to store attention weights for analysis

        # Ensure mlp_ratio is a float for valid dimensionality calculations
        assert isinstance(
            mlp_ratio, float
        ), "MLP ratio should be an integer for valid "
        mlp_dim = int(mlp_ratio * dim) # Calculate the dimension of the MLP layer

        for _ in range(depth):  # Repeat for the specified number of layers
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim,
                                num_heads=heads,
                                qkv_bias=qkv_bias,
                                attn_drop=attn_dropout,
                                proj_drop=dropout,
                            ),
                        ),
                        PreNorm(
                            dim,
                            FeedForward(dim, mlp_dim, dropout_rate=dropout,),
                        )
                        if not revised
                        else FeedForward(
                            dim, mlp_dim, dropout_rate=dropout, revised=True,
                        ),
                    ]
                )
            )

    def forward(self, x):
        self.attn_weights_list = []  # Clear the list of attention weights at the start of each forward pass
        for attn, ff in self.layers: # Obtain both the output and attention weights from the attention layer
            attn_output, attn_weights = attn(x)
            x = attn_output + x  # Apply residual connection
            self.attn_weights_list.append(attn_weights) # Store attention weights for later analysis
            x = ff(x) + x  # Apply feedforward layer with residual connection
        return x, self.attn_weights_list # Return the final output and the list of attention weights