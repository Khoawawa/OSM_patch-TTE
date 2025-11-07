import torch
import torch.nn as nn
import math

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model: int = 256):
        """
        Args:
            d_model (int): The total dimensionality of the output encoding.
                           Must be divisible by 4.
        """
        super().__init__()
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4.")
        
        self.d_model = d_model
        
        # We will split the d_model into 4 parts:
        # d_model / 4 for sin(dx), d_model / 4 for cos(dx)
        # d_model / 4 for sin(dy), d_model / 4 for cos(dy)
        self.d_half = d_model // 2
        self.d_quarter = d_model // 4
        
        # Create the 'div_term' for the denominator: 10000^(2i / (d_model/2))
        # We use d_model/2 because we apply it to dx and dy independently.
        div_term = torch.exp(torch.arange(0, self.d_half, 2).float() * \
                             (-math.log(10000.0) / (self.d_half)))
        
        # Register as a buffer so it's part of the module's state,
        # but not a trainable parameter.
        self.register_buffer('div_term', div_term)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            offsets (torch.Tensor): A tensor of shape (batch_size, 2)
                                    containing the (dx, dy) pairs.

        Returns:
            torch.Tensor: The positional encoding of shape (batch_size, d_model).
        """
        
        # (batch_size, 1) * (d_quarter) -> (batch_size, d_quarter)
        dx_arg = offsets[:, 0:1] * self.div_term
        dy_arg = offsets[:, 1:2] * self.div_term
        
        # Apply sin/cos to dx and dy arguments
        # Each is (batch_size, d_quarter)
        pe_dx_sin = torch.sin(dx_arg)
        pe_dx_cos = torch.cos(dx_arg)
        pe_dy_sin = torch.sin(dy_arg)
        pe_dy_cos = torch.cos(dy_arg)

        # Concatenate all four parts
        # (batch_size, d_quarter * 4) -> (batch_size, d_model)
        pe = torch.cat([pe_dx_sin, pe_dx_cos, pe_dy_sin, pe_dy_cos], dim=1)
        
        return pe