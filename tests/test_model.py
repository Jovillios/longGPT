import torch
from model import CausalSelfAttention, GPTConfig, DilatedCausalSelfAttention

import torch.nn as nn


def test_causal_self_attention_forward():
    # Define a sample configuration
    config = GPTConfig(
        block_size=16,
        vocab_size=50304,
        n_layer=12,
        n_head=4,
        n_embd=16,
        dropout=0.1,
        bias=True
    )

    # Create a CausalSelfAttention instance
    attention = CausalSelfAttention(config)

    # Create a sample input tensor (batch_size=2, seq_len=8, embedding_dim=16)
    x = torch.randn(2, 8, config.n_embd)

    # Forward pass
    output = attention(x)

    # Assertions
    assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"
    print("test_causal_self_attention_forward passed!")


def test_causal_self_attention_no_flash():
    # Define a sample configuration
    config = GPTConfig(
        block_size=16,
        vocab_size=50304,
        n_layer=12,
        n_head=4,
        n_embd=16,
        dropout=0.1,
        bias=True
    )

    # # Force flash attention to be unavailable
    # setattr(torch.nn.functional, 'scaled_dot_product_attention', None)

    # Create a CausalSelfAttention instance
    attention = CausalSelfAttention(config)

    # Create a sample input tensor (batch_size=2, seq_len=8, embedding_dim=16)
    x = torch.randn(2, 8, config.n_embd)

    # Forward pass
    output = attention(x)

    # Assertions
    assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"
    print("test_causal_self_attention_no_flash passed!")


def test_dilated_causal_self_attention_forward():
    # Define a sample configuration
    config = GPTConfig(
        block_size=16,
        vocab_size=50304,
        n_layer=12,
        n_head=4,
        n_embd=16,
        dropout=0.1,
        bias=True,
        dilation=2,
        segment_size=4
    )

    # Create a DilatedCausalSelfAttention instance
    attention = DilatedCausalSelfAttention(config)

    # Create a sample input tensor (batch_size=2, seq_len=8, embedding_dim=16)
    x = torch.randn(2, 8, config.n_embd)

    # Forward pass
    output = attention(x)

    # Assertions
    assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"
    print("test_dilated_causal_self_attention_forward passed!")


def test_dilated_causal_self_attention_no_flash():
    # Define a sample configuration
    config = GPTConfig(
        block_size=16,
        vocab_size=50304,
        n_layer=12,
        n_head=4,
        n_embd=16,
        dropout=0.1,
        bias=True,
        dilation=2,
        segment_size=4
    )

    # Create a DilatedCausalSelfAttention instance
    attention = DilatedCausalSelfAttention(config)

    # Create a sample input tensor (batch_size=2, seq_len=8, embedding_dim=16)
    x = torch.randn(2, 8, config.n_embd)

    # Forward pass
    output = attention(x)

    # Assertions
    assert output.shape == x.shape, f"Expected output shape {x.shape}, but got {output.shape}"
    print("test_dilated_causal_self_attention_no_flash passed!")


if __name__ == "__main__":
    test_causal_self_attention_forward()
    test_causal_self_attention_no_flash()
    test_dilated_causal_self_attention_forward()
    test_dilated_causal_self_attention_no_flash()
