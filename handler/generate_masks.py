import torch

# First, let's modify the preprocess_eeg_data function to also create the necessary masks.


def generate_masks_from_embeddings(embeddings: torch.Tensor, valid_length: int) -> (torch.Tensor, torch.Tensor):
    """
    Generate attention masks and their inverse for a given embeddings tensor.

    Parameters:
    - embeddings (torch.Tensor): The embeddings tensor.
    - valid_length (int): The length of valid (non-padded) embeddings.

    Returns:
    - tuple: A tuple containing the attention mask and its inverse.
    """
    attn_mask = torch.zeros(embeddings.size(0))
    attn_mask[:valid_length] = 1.0
    attn_mask_invert = 1.0 - attn_mask
    return attn_mask, attn_mask_invert

