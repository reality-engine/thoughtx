import torch

# First, let's modify the preprocess_eeg_data function to also create the necessary masks.

def generate_masks_from_embeddings(embeddings: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """
    Generate attention masks and their inverse for a given embeddings tensor.

    Parameters:
    - embeddings (torch.Tensor): The embeddings tensor.

    Returns:
    - tuple: A tuple containing the attention mask and its inverse.
    """
    # Assuming non-zero embeddings represent valid tokens and zeros represent padding
    attn_mask = (embeddings.sum(dim=-1) != 0).float()
    attn_mask_invert = 1.0 - attn_mask
    return attn_mask, attn_mask_invert