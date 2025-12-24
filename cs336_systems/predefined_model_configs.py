"""Predefined model configs."""

from cs336_basics import transformer


def _get_small_model_config(
    vocab_size: int, max_context_length: int
) -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=12,
        num_heads=12,
        d_model=768,
        d_ff=3072,
        rope_theta=10000.0,
        context_length=max_context_length,
    )


def _get_medium_model_config(
    vocab_size: int, max_context_length: int
) -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=24,
        num_heads=16,
        d_model=1024,
        d_ff=4096,
        rope_theta=10000.0,
        context_length=max_context_length,
    )


def _get_large_model_config(
    vocab_size: int, max_context_length: int
) -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=36,
        num_heads=20,
        d_model=1280,
        d_ff=5120,
        rope_theta=10000.0,
        context_length=max_context_length,
    )


def _get_xl_model_config(
    vocab_size: int, max_context_length: int
) -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=48,
        num_heads=25,
        d_model=1600,
        d_ff=6400,
        rope_theta=10000.0,
        context_length=max_context_length,
    )


def _get_2p7b_model_config(
    vocab_size: int, max_context_length: int
) -> transformer.TransformerConfig:
    return transformer.TransformerConfig(
        vocab_size=vocab_size,
        num_layers=32,
        num_heads=32,
        d_model=2560,
        d_ff=10240,
        rope_theta=10000.0,
        context_length=max_context_length,
    )


def get_predefined_model_configs(
    vocab_size: int, max_context_length: int
) -> dict[str, transformer.TransformerConfig]:
    """Get predefined model configs."""
    return {
        "small": _get_small_model_config(vocab_size, max_context_length),
        "medium": _get_medium_model_config(vocab_size, max_context_length),
        "large": _get_large_model_config(vocab_size, max_context_length),
        "xl": _get_xl_model_config(vocab_size, max_context_length),
        "2p7B": _get_2p7b_model_config(vocab_size, max_context_length),
    }
