import torch.nn as nn

from .patch_embed import EmbeddingStem
from .transformer import Transformer
from .modules import OutputLayer

# Definition of the Vision Transformer class
class VisionTransformer(nn.Module):
    def __init__(
        self,
        image_size=224, # Input image size
        patch_size=16, # Size of the patches to be extracted from the input images
        in_channels=1, # Number of input channels (e.g., 1 for grayscale, 3 for RGB)
        embedding_dim=768, # Dimensionality of the token embeddings
        num_layers=12, # Number of transformer layers
        num_heads=12, # Number of attention heads in each transformer layer
        qkv_bias=True, # Whether to include a bias term in the query, key, value projections
        mlp_ratio=4.0, # Ratio of the hidden dimension of the MLP layers to the embedding dimension
        use_revised_ffn=True, # Use revised feedforward network
        dropout_rate=0.0, # Dropout rate
        attn_dropout_rate=0.0, # Attention dropout rate
        use_conv_stem=True, # Use a convolutional stem before patch embedding
        use_conv_patch=False, # Use convolutional patches
        use_linear_patch=False, # Use linear patch projection
        use_conv_stem_original=True, # Use the original convolutional stem design
        use_stem_scaled_relu=False, # Use scaled ReLU in the stem
        hidden_dims=None, # Hidden dimensions for the convolutional stem, if any
        cls_head=False, # Include a classification head
        num_classes=1000, # Number of classes for classification
        representation_size=None, # Size of the representation layer, if any
    ):
        super(VisionTransformer, self).__init__()

        # Initialization of the embedding layer
        self.embedding_layer = EmbeddingStem(
            image_size=image_size,
            patch_size=patch_size,
            channels=in_channels,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            conv_patch=use_conv_patch,
            linear_patch=use_linear_patch,
            conv_stem=use_conv_stem,
            conv_stem_original=use_conv_stem_original,
            conv_stem_scaled_relu=use_stem_scaled_relu,
            position_embedding_dropout=dropout_rate,
            cls_head=cls_head,
        )

        # Initialization of the transformer layer
        self.transformer = Transformer(
            dim=embedding_dim,
            depth=num_layers,
            heads=num_heads,
            mlp_ratio=mlp_ratio,
            attn_dropout=attn_dropout_rate,
            dropout=dropout_rate,
            qkv_bias=qkv_bias,
            revised=use_revised_ffn,
        )

        self.post_transformer_ln = nn.LayerNorm(embedding_dim) # Layer normalization following the transformer

        # Initialization of the output (classification) layer
        self.cls_layer = OutputLayer(
            embedding_dim,
            num_classes=num_classes,
            representation_size=representation_size,
            cls_head=cls_head,
        )

    def forward(self, x):
        # Pass input x through the embedding layer
        x = self.embedding_layer(x)
        # Pass the output of the embedding layer through the transformer
        x, attn_weights = self.transformer(x)
        # Apply layer normalization to the output of the transformer
        x = self.post_transformer_ln(x)
        # Pass the normalized output through the classification layer
        x = self.cls_layer(x)
        # Return the final output
        return x