import torch
from torch import Tensor
from jaxtyping import Int
from einops import pack


def format_green_truecolor(text: str, alpha: float) -> str:
    alpha = max(0.0, min(1.0, alpha))
    # Green channel from 0 (black) to 255 (green)
    g_value = int(round(alpha * 255))
    
    # 48;2;R;G;B sets the background color in truecolor
    # 0;R;G;B sets the foreground color (not used here)
    # We'll keep R=0 and B=0, and just vary G
    color_code = f"\033[48;2;{255 - g_value};255;{255 - g_value}m"
    reset_code = "\033[0m"
    
    return f"{color_code}{text}{reset_code}"


def render_expressions(tokens: list[str], expressions: list[float]) -> str:
    # Expressions are in the range [0, 1]
    assert len(tokens) == len(expressions)
    colored_tokens = [
        format_green_truecolor(token, expression)
        for token, expression in zip(tokens, expressions)
    ]
    return ''.join(colored_tokens) + '\n'


def last_subseq_idx(
    sequence: Int[Tensor, "seq_len"], 
    subsequence: Int[Tensor, "sub_len"]
) -> int:
    sub_length = subsequence.size(0)

    # Create a sliding window of sub_length over the sequence
    windows = sequence.unfold(0, sub_length, 1)

    # Compare each window with the subsequence
    matches = (windows == subsequence).all(dim=1)

    # Find the last match
    indices = torch.nonzero(matches, as_tuple=True)[0]

    return indices[-1].item() if indices.numel() > 0 else -1
    

def chunk_sampled_indices(
    length: int,
    chunk_n: int,
    sample_n_per_chunk: int,
) -> Int[Tensor, "chunk sample"]:
    """Chunks the "length" range into `chunk_n` chunks, and sample `sample_n_per_chunk`
    indices from each chunk. The indices are in the range [0, length)."""

    chunk_size = length // chunk_n
    chunks = [
        torch.randperm(chunk_size)[:sample_n_per_chunk] + (i * chunk_size) 
        for i in range(chunk_n)
    ]
    return pack(chunks, '* sample')[0]


# Adapted from tether/tether/core/encoder.py.
def convert_to_byte_array(s: str) -> bytearray:
    byte_array = bytearray()
    assert s.startswith("bytes:"), s
    s = s[6:]
    while len(s) > 0:
        if s[0] == "\\":
            # Hex encoding.
            assert s[1] == "x"
            assert len(s) >= 4
            byte_array.append(int(s[2:4], 16))
            s = s[4:]
        else:
            # Regular ascii encoding.
            byte_array.append(ord(s[0]))
            s = s[1:]
    return byte_array


def drop_indices(
    all_indices: Int[Tensor, "L"],
    indices_to_drop: Int[Tensor, "l"],
    min_length: int | None = None,
    assume_unique: bool = True,
) -> Int[Tensor, "N"]:
    """Drop a set of indices, but ensure that the returned tensor has at least
    `min_length` indices.
    
    If the resulting tensor after dropping indices has less than `min_length`
    elements, we will raise an error.
    
    If `assume_unique` is True, we assume both input tensors contain unique
    elements."""
    mask: Int[Tensor, "L"] = torch.isin(
        all_indices, indices_to_drop, invert=True, assume_unique=assume_unique
    )
    filtered_indices: Int[Tensor, "n"] = all_indices[mask]
    n = len(filtered_indices)
    if min_length is not None and n < min_length:
        raise ValueError('Not enough indices after dropping.')
    return filtered_indices
