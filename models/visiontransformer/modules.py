import torch
import torch.nn as nn


class PreNorm(nn.Module): # Pre-normalization class
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.norm = nn.LayerNorm(dim)  # Initialize layer normalization
        self.fn = fn  # Save the function to be applied after normalization

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)  # Apply normalization then the function fn


class Attention(nn.Module):  # Self-attention mechanism class
    def __init__(
        self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0
    ):
        super(Attention, self).__init__()

        # Ensure the embedding dimension is divisible by the number of heads
        assert (
            dim % num_heads == 0
        ), "Embedding dimension should be divisible by number of heads"

        self.num_heads = num_heads  # Save the number of heads
        head_dim = dim // num_heads  # Dimension per head
        self.scale = head_dim ** -0.5  # Scaling factor for attention scores

        # Linear layers for query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)  # Attention dropout
        self.proj = nn.Linear(dim, dim)  # Linear layer for final projection
        self.proj_drop = nn.Dropout(proj_drop)  # Projection dropout
        self.attn_weights = None # Store attention weights

    def forward(self, x):
        B, N, C = x.shape  # Batch size B, number of tokens N, channel size C
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        # Unpack query, key, value
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale # Calculate attention scores

        attn = attn.softmax(dim=-1)  # Apply softmax

        self.attn_weights = attn # Store attention weights for visualization or analysis
        attn = self.attn_drop(attn) # Apply dropout
        

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # Apply attention and reshape
        x = self.proj(x)  # Apply final projection
        x = self.proj_drop(x)  # Apply dropout
        return x, self.attn_weights # Return output and attention weights


class FeedForward(nn.Module):
    """
    Implementing the feedforward network of the Transformer
    """

    def __init__(self, dim, hidden_dim, dropout_rate=0.0, revised=False):
        super(FeedForward, self).__init__()
        if not revised:
            # Sequence of layers for the original feedforward network
            """
            Original: https://arxiv.org/pdf/2010.11929.pdf
            """
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Linear(hidden_dim, dim),
            )
        else:
            # Sequence of layers for the revised feedforward network with scaled ReLU
            self.net = nn.Sequential(
                nn.Conv1d(dim, hidden_dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout_rate),
                nn.Conv1d(hidden_dim, dim, kernel_size=1, stride=1),
                nn.BatchNorm1d(dim),
                nn.GELU(),
            )

        self.revised = revised # Save the revised flag
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        # Initialize weights of linear layers
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.bias, std=1e-6)  # バイアスの初期化

    def forward(self, x):
        # Process input through the network
        if self.revised:
            # Process input through the network
            x = x.permute(0, 2, 1)
            x = self.net(x)
            x = x.permute(0, 2, 1)
        else:
            # For original network, directly apply linear layers
            x = self.net(x)

        return x


class OutputLayer(nn.Module):  # Class implementing the output layer
    def __init__(
        self,
        embedding_dim,  # Embedding dimension
        num_classes=1000,  # Number of classes for classification
        representation_size=None,  # Size of representation layer
        cls_head=False,  # Flag to use a class head
    ):
        super(OutputLayer, self).__init__()

        self.num_classes = num_classes # Save the number of classes
        modules = []
        if representation_size:
            # If representation size is specified, create layers accordingly
            modules.append(nn.Linear(embedding_dim, representation_size))
            modules.append(nn.Tanh())
            modules.append(nn.Linear(representation_size, num_classes))
        else:
            # If no representation size, create a single linear layer
            modules.append(nn.Linear(embedding_dim, num_classes))

        self.net = nn.Sequential(*modules)  # Create a sequential layer

        if cls_head:
            self.to_cls_token = nn.Identity()  # Identity layer if class head is present

        self.cls_head = cls_head  # Save the class head flag
        self.num_classes = num_classes  # Save the number of classes
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        # Initialize weights of linear layers
        for name, module in self.net.named_children():
            if isinstance(module, nn.Linear):
                # Initialize weights for the linear layer related to classification
                if module.weight.shape[0] == self.num_classes:
                    nn.init.zeros_(module.weight)
                    nn.init.zeros_(module.bias)

    def forward(self, x):
        # Process input through the layer sequence
        if self.cls_head:
            # Process using class head if present
            x = self.to_cls_token(x[:, 0])
        else:
            """
            Scaling Vision Transformer: https://arxiv.org/abs/2106.04560
            """
            # Average pooling across tokens if no class head is used
            x = torch.mean(x, dim=1)

        return self.net(x) # Return the output of the layer sequence