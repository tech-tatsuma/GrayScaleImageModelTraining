import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from .utils import trunc_normal_


def pair(t):
    return t if isinstance(t, tuple) else (t, t)  # Utility function to ensure the input is a tuple


class EmbeddingStem(nn.Module):  # Definition of the EmbeddingStem class
    def __init__(
        self,
        image_size=224,  # Default image size
        patch_size=16,  # Default patch size
        channels=3,  # Default number of channels (e.g., RGB)
        embedding_dim=768, # Default embedding dimension
        hidden_dims=None,  # Dimensions of hidden layers, if any
        conv_patch=False, # Flag to use convolutional patches
        linear_patch=False,  # Flag to use linear patches
        conv_stem=True,  # Flag to use a convolutional stem
        conv_stem_original=True,  # Flag to use the original convolutional stem
        conv_stem_scaled_relu=False,  # Flag to use scaled ReLU in convolutional stem
        position_embedding_dropout=None,  # Dropout rate for position embeddings
        cls_head=True,  # Flag to use a class token head
    ):
        super(EmbeddingStem, self).__init__() 

        # Ensure only one mode is active at a time
        assert (
            sum([conv_patch, conv_stem, linear_patch]) == 1
        ), "Only one of three modes should be active"

        # Convert image and patch sizes to tuples
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Ensure image dimensions are divisible by patch size
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        # Check for incompatible options
        assert not (
            conv_stem and cls_head
        ), "Cannot use [CLS] token approach with full conv stems for ViT"

        # Initialize class token, position embeddings, and dropout if using linear or conv patches
        if linear_patch or conv_patch:
            self.grid_size = (
                image_height // patch_height,
                image_width // patch_width,
            )
            num_patches = self.grid_size[0] * self.grid_size[1]

            if cls_head:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
                num_patches += 1

            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embedding_dim)
            )
            self.pos_drop = nn.Dropout(p=position_embedding_dropout)

        # Initialize projection for conv_patch
        if conv_patch:
            self.projection = nn.Sequential(
                nn.Conv2d(
                    channels,
                    embedding_dim,
                    kernel_size=patch_size,
                    stride=patch_size,
                ),
            )
        # Initialize projection for linear_patch
        elif linear_patch:
            patch_dim = channels * patch_height * patch_width
            self.projection = nn.Sequential(
                Rearrange(
                    'b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                    p1=patch_height,
                    p2=patch_width,
                ),
                nn.Linear(patch_dim, embedding_dim),
            )
        # Initialize projection for conv_stem
        elif conv_stem:
            assert (
                conv_stem_scaled_relu ^ conv_stem_original
            ), "Can use either the original or the scaled relu stem"

            if not isinstance(hidden_dims, list):
                raise ValueError("Cannot create stem without list of sizes")

            # Initialize original convolutional stem
            if conv_stem_original:
                """
                Conv stem from https://arxiv.org/pdf/2106.14881.pdf
                """

                hidden_dims.insert(0, channels)
                modules = []
                for i, (in_ch, out_ch) in enumerate(
                    zip(hidden_dims[:-1], hidden_dims[1:])
                ):
                    modules.append(
                        nn.Conv2d(
                            in_ch,
                            out_ch,
                            kernel_size=3,
                            stride=2 if in_ch != out_ch else 1,
                            padding=1,
                            bias=False,
                        ),
                    )
                    modules.append(nn.BatchNorm2d(out_ch),)
                    modules.append(nn.ReLU(inplace=True))

                modules.append(
                    nn.Conv2d(
                        hidden_dims[-1], embedding_dim, kernel_size=1, stride=1,
                    ),
                )
                self.projection = nn.Sequential(*modules)

            # Initialize scaled ReLU convolutional stem
            elif conv_stem_scaled_relu:
                """
                Conv stem from https://arxiv.org/pdf/2109.03810.pdf
                """
                assert (
                    len(hidden_dims) == 1
                ), "Only one value for hidden_dim is allowed"
                mid_ch = hidden_dims[0]

                # Set flags
                self.projection = nn.Sequential(
                    nn.Conv2d(
                        channels, mid_ch,
                        kernel_size=7, stride=2, padding=3, bias=False,
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mid_ch, mid_ch,
                        kernel_size=3, stride=1, padding=1, bias=False,
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mid_ch, mid_ch,
                        kernel_size=3, stride=1, padding=1, bias=False,
                    ),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        mid_ch, embedding_dim,
                        kernel_size=patch_size // 2, stride=patch_size // 2,
                    ),
                )
                # fmt: on

            else:
                raise ValueError("Undefined convolutional stem type defined")

        # Set flags
        self.conv_stem = conv_stem
        self.conv_patch = conv_patch
        self.linear_patch = linear_patch
        self.cls_head = cls_head 

        self._init_weights()  # Initialize weights

    def _init_weights(self):
        if not self.conv_stem:
            trunc_normal_(self.pos_embed, std=0.02)  # Initialize weights for position embeddings

    def forward(self, x):
        # Process input through the projection based on the mode
        if self.conv_stem:
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)
            return x

        elif self.linear_patch:
            x = self.projection(x)
        elif self.conv_patch:
            x = self.projection(x)
            x = x.flatten(2).transpose(1, 2)

        # Add class token if cls_head is true
        if self.cls_head:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        # Apply position embedding dropout and return the result
        return self.pos_drop(x + self.pos_embed) 